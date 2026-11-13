# -*- coding: utf-8 -*-
"""
train.py —— 训练例程

两种协议:
  A. 仿真从头训练(主协议, 杜绝真实数据重用):
     流式无限仿真集训练, 固定仿真验证集上早停+选型, 真实数据零接触;
  B. 真实数据分组微调(对照组):
     仅用 Train9 微调、仅用 Val3 早停选型; Test3 全程不参与, 留最终盲测。
"""

import copy
import queue
import threading
import time
import numpy as np
import torch

import hconfig as H
from models import ConformerResUNet1D, UnifiedInversionLoss
from baselines import spectrum_metrics, pearson_r


def _start_prefetcher(stream, steps, look=2):
    """后台线程按 step 顺序预生成训练 batch(双缓冲)。
    batch(step) 是由 step 决定的纯函数, 顺序产出与同步生成完全一致(可复现);
    numpy 矩阵运算释放 GIL, 可与 torch 训练真正并行, 消除数据生成串行等待。"""
    q: "queue.Queue" = queue.Queue(maxsize=look)

    def producer():
        try:
            for step in range(1, steps + 1):
                q.put(stream.batch(step))
        finally:
            q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    return q


def build_model(cfg):
    return ConformerResUNet1D(
        in_channels=1,
        out_bins=128,
        stem_dim=cfg.get("stem_dim", 32),
        d_model=cfg.get("d_model", 128),
        num_heads=cfg.get("num_heads", 4),
        num_conformer_layers=cfg.get("num_layers", 3),
        dropout=cfg.get("dropout", 0.05),
    )


@torch.no_grad()
def predict_spectra(model, echoes, device, chunk=64):
    """分块推理: 始终使用与训练一致的批形状, 避免新形状触发不稳定 kernel。"""
    model.eval()
    arr = np.asarray(echoes, dtype=np.float32)
    out = []
    for i in range(0, len(arr), chunk):
        xs = torch.tensor(arr[i:i + chunk], device=device).unsqueeze(1)
        out.append(model(xs).cpu().numpy())
    return np.concatenate(out, axis=0)


def composite_score(mean_r, mean_peak_err, mean_fwd_rmse):
    """与旧 m6 相同权重的综合分(相关度60/峰位25/前向15), 便于历史对照。"""
    return (mean_r * 60.0
            + max(0.0, (1.0 - mean_peak_err * 2.0)) * 25.0
            + max(0.0, (1.0 - min(mean_fwd_rmse, 1.0) * 5.0)) * 15.0)


def eval_model_on_set(model, echoes, specs, device, masks=None):
    preds = predict_spectra(model, echoes, device)
    ms = [spectrum_metrics(p, s, e, None if masks is None else masks[i])
          for i, (p, s, e) in enumerate(zip(preds, specs, echoes))]
    mean_r = float(np.mean([m["pearson_r"] for m in ms]))
    mean_pk = float(np.mean([m["peak_err_log"] for m in ms]))
    mean_fw = float(np.mean([m["fwd_rmse"] for m in ms]))
    return {
        "mean_r": mean_r,
        "mean_peak_err": mean_pk,
        "mean_fwd_rmse": mean_fw,
        "score": composite_score(mean_r, mean_pk, mean_fw),
        "preds": preds,
    }


def train_sim_trial(cfg, stream, val_echoes, val_specs, device,
                    steps, val_every, patience, log_fn, seed=0, val_masks=None):
    """协议A: 仿真流式训练, 固定仿真验证集早停+选型。返回 (best_state, 纪录)。
    stream.batch 可返回 (x,y) 或 (x,y,valid_mask) 三元组(定稿v3扩展分布);
    val_masks 为验证集截断 mask 时, 前向指标只算有效窗。"""
    torch.manual_seed(seed)
    model = build_model(cfg).to(device)
    criterion = UnifiedInversionLoss(
        lambda_huber=cfg.get("lambda_huber", 1.0),
        lambda_wass=cfg.get("lambda_wass", 0.5),
        lambda_phys=cfg.get("lambda_phys", 0.3),
        lambda_sparse=cfg.get("lambda_sparse", 0.01),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.get("lr", 5e-4), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-5)

    best_score, best_state, bad_evals = -1e18, None, 0
    t0 = time.time()
    running = []
    nan_skips = 0
    # GPU 占空比节流: 有效负载恒定 ≤ GPU_DUTY(用户要求 ≤70% 防黑屏)。
    # 每 DUTY_PERIOD 步统计纯计算耗时, 按比例补 sleep。
    duty = H.GPU_DUTY if device.type == "cuda" else 1.0
    duty_win_t0 = time.time()
    DUTY_PERIOD = 10
    slept_total = 0.0

    batch_q = _start_prefetcher(stream, steps)
    for step in range(1, steps + 1):
        model.train()
        item = batch_q.get()
        if item is None:   # 生产者异常终止(理论上不会发生)
            break
        if len(item) == 3:
            bx, by, bmask = item
            mask_t = torch.tensor(bmask, device=device)
        else:
            bx, by = item
            mask_t = None
        bx = torch.tensor(bx, device=device).unsqueeze(1)
        by = torch.tensor(by, device=device)
        optimizer.zero_grad()
        pred = model(bx)
        loss, _ = criterion(pred, by, bx, mask_t)
        if not torch.isfinite(loss):
            nan_skips += 1
            optimizer.zero_grad()
            if nan_skips % 10 == 1:
                log_fn(f"    step {step}: loss 非有限, 跳过该批 (累计 {nan_skips})")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()
        running.append(float(loss.detach()))

        if duty < 1.0 and step % DUTY_PERIOD == 0:
            compute = time.time() - duty_win_t0
            sleep_for = compute * (1.0 / duty - 1.0)
            if sleep_for > 0:
                time.sleep(sleep_for)
                slept_total += sleep_for
            duty_win_t0 = time.time()

        if step % 500 == 0:
            log_fn(f"    step {step:>6d}/{steps} | loss(500均值) "
                   f"{np.mean(running[-500:]):.5f} | lr {scheduler.get_last_lr()[0]:.2e} "
                   f"| 用时 {time.time() - t0:.0f}s")

        if step % val_every == 0 or step == steps:
            ev = eval_model_on_set(model, val_echoes, val_specs, device,
                                   masks=val_masks)
            log_fn(f"    [验证@{step:>6d}] r={ev['mean_r']:.4f} "
                   f"峰位误差={ev['mean_peak_err']:.4f} 前向RMSE={ev['mean_fwd_rmse']:.5f} "
                   f"综合分={ev['score']:.2f}")
            if ev["score"] > best_score + 1e-6:
                best_score = ev["score"]
                best_state = copy.deepcopy(model.state_dict())
                bad_evals = 0
            else:
                bad_evals += 1
                if bad_evals >= patience:
                    log_fn(f"    早停于 step {step} (连续 {patience} 次无提升)")
                    break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    log_fn(f"    完成: 最佳仿真验证综合分 {best_score:.2f}, 总用时 {time.time() - t0:.0f}s, "
           f"非有限批累计 {nan_skips}, 负载节流休眠 {slept_total:.0f}s"
           + (f" (占空比≤{duty:.0%})" if duty < 1.0 else ""))
    return best_state, best_score


def finetune_real(cfg, init_state, records, device, log_fn, seed=0,
                  max_epochs=400, eval_every=10, patience=12, lr_ft=5e-5):
    """协议B: 仅 Train9 微调, 仅 Val3 早停/选型。Test3 不进入本函数。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(cfg).to(device)
    model.load_state_dict(init_state)
    criterion = UnifiedInversionLoss(
        lambda_huber=cfg.get("lambda_huber", 1.0),
        lambda_wass=cfg.get("lambda_wass", 0.5),
        lambda_phys=cfg.get("lambda_phys", 0.3),
        lambda_sparse=cfg.get("lambda_sparse", 0.01),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_ft, weight_decay=1e-4)

    tr = [r for r in records if r["split"] == "Train9"]
    va = [r for r in records if r["split"] == "Val3"]
    tr_x = torch.tensor(np.stack([r["echo"] for r in tr]),
                        device=device).unsqueeze(1)
    tr_y = torch.tensor(np.stack([r["spec"] for r in tr]), device=device)
    # v3协议: 截断记录(1800回波)的物理前向损失只算有效时间窗
    tr_mask = torch.tensor(np.stack(
        [r.get("valid_mask", np.ones(1024, np.float32)) for r in tr]),
        device=device)
    va_e = [r["echo"] for r in va]
    va_s = [r["spec"] for r in va]
    va_m = [r.get("valid_mask") for r in va]
    va_m = np.stack(va_m) if all(m is not None for m in va_m) else None

    best_score, best_state, bad = -1e18, None, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(tr_x)
        loss, _ = criterion(pred, tr_y, tr_x, tr_mask)
        if not torch.isfinite(loss):
            log_fn(f"    [微调 epoch {ep}] loss 非有限, 跳过")
            optimizer.zero_grad()
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        if ep % eval_every == 0 or ep == max_epochs:
            ev = eval_model_on_set(model, va_e, va_s, device, masks=va_m)
            log_fn(f"    [微调 epoch {ep:>4d}] Val3 r={ev['mean_r']:.4f} "
                   f"峰位={ev['mean_peak_err']:.4f} 综合分={ev['score']:.2f} "
                   f"(仅Val3, Test3未参与)")
            if ev["score"] > best_score + 1e-6:
                best_score = ev["score"]
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    log_fn(f"    微调早停于 epoch {ep}")
                    break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    return best_state, best_score
