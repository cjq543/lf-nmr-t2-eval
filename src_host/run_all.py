# -*- coding: utf-8 -*-
"""
run_all.py —— 上位机大模型 v2 无人值守总控(子进程隔离版)

阶段流水(断点续跑, 每阶段完成即落盘 run_state.json):
  1 simdata     固定仿真验证/测试集生成与缓存(种子互不相交)
  2 params      传统方法参数冻结(NNLS 正则、SG 窗口, 仅开发集选定)
  3 trials      仿真从头训练 6 组超参 trial(子进程: 先GPU, 崩溃自动回退CPU)
  4 seeds       最优配置 3 种子复训(报告种子方差)
  5 finetune    真实数据协议B微调: Train9 训练 / Val3 早停 / Test3 不接触(CPU)
  6 eval        冻结盲测(CPU): 仿真测试集(真值) + 真实15记录(厂商参考)
                + 旧m6只读对照 + 低信噪比压力子集 + 最终报告

稳定性设计(本机实测 RTX 2060 + 546.33 驱动对个别 kernel 形状存在 CUDA 非法访存,
conv/batchnorm/matmul 稳定、注意力等个别算子故障):
  * 所有 GPU 训练都放在独立子进程(trial_worker.py), 崩溃仅损失该子进程;
  * 父进程自身只在 CPU 上做微调/推理/评估, 永不创建 CUDA 上下文;
  * 推理按固定块大小分批, 避免新形状。

用法: python src/run_all.py [--smoke]
"""

import json
import os
import subprocess
import sys
import time
import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hconfig as H
from simulator import generate_dataset
from realdata import load_real_records
from models import ConformerResUNet1D
from baselines import select_classical_params
from train import build_model, finetune_real
import evaluate as EV

SMOKE = "--smoke" in sys.argv
HAS_CUDA = torch.cuda.is_available() and not SMOKE
WORKER = os.path.join(H.SRC_DIR, "trial_worker.py")

if SMOKE:
    GPU_STEPS, CPU_STEPS, GPU_TIMEOUT, CPU_TIMEOUT = 300, 200, 600, 600
else:
    GPU_STEPS, CPU_STEPS = H.TRAIN_STEPS_GPU, H.CPU_STEPS
    GPU_TIMEOUT, CPU_TIMEOUT = H.TRIAL_TIMEOUT_GPU, H.TRIAL_TIMEOUT_CPU


def log_fn(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(H.MAIN_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_state():
    if os.path.exists(H.STATE_FILE):
        with open(H.STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": []}


def save_state(st):
    tmp = H.STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, H.STATE_FILE)


def stage(name):
    def deco(fn):
        def run(*a, **kw):
            if name in load_state()["completed"]:
                log_fn(f"== 阶段 {name} 已完成, 跳过 ==")
                return None
            log_fn(f"==== 阶段 {name} 开始 ====")
            t0 = time.time()
            out = fn(*a, **kw)
            st = load_state()
            st["completed"].append(name)
            st[f"{name}_seconds"] = round(time.time() - t0, 1)
            save_state(st)
            log_fn(f"==== 阶段 {name} 完成, 用时 {time.time() - t0:.0f}s ====")
            return out
        return run
    return deco


def _safe_remove(p):
    try:
        if os.path.exists(p):
            os.remove(p)
    except OSError:
        pass


def run_trial_worker(cfg, seed, ckpt, rec_p, log, kind="sim", init_ckpt=None):
    """在子进程中跑一个训练任务: 先 GPU, 失败/超时自动回退 CPU。
    成功判据: 子进程退出码 0 且 ckpt/json 双双产出。"""
    cfg_file = rec_p + ".cfg.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)

    def attempt(device, steps, timeout):
        cmd = [sys.executable, WORKER,
               "--kind", kind, "--config", cfg_file, "--device", device,
               "--steps", str(steps), "--seed", str(seed),
               "--out-ckpt", ckpt, "--out-json", rec_p]
        if init_ckpt:
            cmd += ["--init-ckpt", init_ckpt]
        log(f"    -> 子进程训练: device={device} steps={steps} seed={seed}")
        try:
            r = subprocess.run(cmd, timeout=timeout, capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               cwd=H.ROOT_DIR)
        except subprocess.TimeoutExpired:
            log(f"    [超时] device={device} 超过 {timeout}s, 清理产物")
            _safe_remove(ckpt)
            _safe_remove(rec_p)
            return False
        if r.returncode != 0 or not (os.path.exists(ckpt) and os.path.exists(rec_p)):
            tail = (r.stderr or r.stdout or "")[-700:].strip()
            log(f"    [失败] device={device} exit={r.returncode}\n{tail}")
            _safe_remove(ckpt)
            _safe_remove(rec_p)
            return False
        return True

    if HAS_CUDA:
        if attempt("cuda", GPU_STEPS, GPU_TIMEOUT):
            return True
        log("    GPU 子进程失败, 自动回退 CPU 重跑同一任务")
    return attempt("cpu", CPU_STEPS, CPU_TIMEOUT)


# ---------------------------------------------------------------- 超参矩阵
TRIAL_CONFIGS = [
    {"tag": "T01_基线均衡", "stem_dim": 32, "d_model": 128, "num_heads": 4, "num_layers": 3,
     "lambda_huber": 1.0, "lambda_wass": 0.5, "lambda_phys": 0.3, "lr": 5e-4},
    {"tag": "T02_峰位强化", "stem_dim": 32, "d_model": 128, "num_heads": 4, "num_layers": 3,
     "lambda_huber": 1.0, "lambda_wass": 0.8, "lambda_phys": 0.3, "lr": 5e-4},
    {"tag": "T03_前向强化", "stem_dim": 32, "d_model": 128, "num_heads": 4, "num_layers": 3,
     "lambda_huber": 1.0, "lambda_wass": 0.5, "lambda_phys": 0.6, "lr": 4e-4},
    {"tag": "T04_深宽架构", "stem_dim": 40, "d_model": 160, "num_heads": 8, "num_layers": 4,
     "lambda_huber": 1.0, "lambda_wass": 0.6, "lambda_phys": 0.4, "lr": 3.5e-4},
    {"tag": "T05_紧凑高速", "stem_dim": 24, "d_model": 96, "num_heads": 2, "num_layers": 2,
     "lambda_huber": 1.0, "lambda_wass": 0.4, "lambda_phys": 0.3, "lr": 6e-4},
    {"tag": "T06_退火长训", "stem_dim": 32, "d_model": 128, "num_heads": 4, "num_layers": 4,
     "lambda_huber": 1.2, "lambda_wass": 0.7, "lambda_phys": 0.5, "lr": 3e-4},
]

REAL_DIR_README = "真实数据只读: 微调仅Train9 / 早停仅Val3 / Test3盲测"
DEVICE = torch.device("cpu")   # 父进程一律 CPU, CUDA 全部走子进程


def _best_trial():
    trials_dir = os.path.join(H.RESULT_DIR, "trials")
    best = None
    for fn in os.listdir(trials_dir):
        if not fn.endswith(".json") or fn.endswith(".cfg.json"):
            continue
        with open(os.path.join(trials_dir, fn), encoding="utf-8") as f:
            d = json.load(f)
        if "sim_val_score" not in d:
            continue
        if best is None or d["sim_val_score"] > best["sim_val_score"]:
            best = d
    return best


def main():
    log_fn("=" * 72)
    log_fn("上位机大模型 v2 无人值守训练启动 (统一协议·无数据重用·多分支对照)")
    log_fn(f"输出根目录: {H.ROOT_DIR}  (SMOKE={SMOKE})")
    log_fn(f"GPU可用={HAS_CUDA}(训练走子进程) | 预算: GPU {GPU_STEPS}步 / "
           f"CPU回退 {CPU_STEPS}步 | 评估固定CPU")
    log_fn("=" * 72)

    @stage("simdata")
    def s_simdata():
        cache = os.path.join(H.RESULT_DIR, "sim_cache.npz")
        if os.path.exists(cache):
            return None
        t0 = time.time()
        vx, vy, _ = generate_dataset(H.SIM_VAL_N, H.SIM_VAL_SEED)
        tx, ty, _ = generate_dataset(H.SIM_TEST_N, H.SIM_TEST_SEED)
        lx, ly, _ = generate_dataset(192, H.SIM_TEST_SEED + 7777, snr_filter="low")
        np.savez_compressed(cache, vx=vx, vy=vy, tx=tx, ty=ty, lx=lx, ly=ly)
        log_fn(f"  仿真数据集: 验证{vx.shape} 测试{tx.shape} 低信噪比{lx.shape}, "
               f"生成用时 {time.time() - t0:.0f}s")
        return None

    @stage("params")
    def s_params():
        out = os.path.join(H.RESULT_DIR, "classical_params.json")
        if os.path.exists(out):
            return None
        cache = np.load(os.path.join(H.RESULT_DIR, "sim_cache.npz"))
        n_sel = 96 if SMOKE else len(cache["vx"])   # 全部 1024 条验证集
        win, poly, lam_sg, lam_direct = select_classical_params(
            cache["vx"][:n_sel], cache["vy"][:n_sel], log_fn)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"sg_win": int(win), "sg_poly": int(poly),
                       "nnls_lambda": float(lam_sg),
                       "nnls_lambda_direct": float(lam_direct)},
                      f, ensure_ascii=False, indent=1)
        return None

    @stage("trials")
    def s_trials():
        os.makedirs(os.path.join(H.RESULT_DIR, "trials"), exist_ok=True)
        for cfg in TRIAL_CONFIGS:
            tag = cfg["tag"]
            ckpt = os.path.join(H.MODEL_DIR, f"h1_{tag}.pth")
            rec_p = os.path.join(H.RESULT_DIR, "trials", f"{tag}.json")
            if os.path.exists(ckpt) and os.path.exists(rec_p):
                with open(rec_p, encoding="utf-8") as f:
                    d = json.load(f)
                log_fn(f"  已存在, 跳过: {tag} (分 {d['sim_val_score']:.2f})")
                continue
            log_fn(f"  -> Trial {tag}: {cfg}")
            ok = run_trial_worker(cfg, 0, ckpt, rec_p, log_fn)
            if not ok:
                # 防呆: 双设备尝试后仍无产物必须中止, 绝不带着缺口"假完成"
                raise RuntimeError(
                    f"Trial {tag} 在 GPU 与 CPU 尝试后均失败, 阶段中止(下次运行重试)")
        best = _best_trial()
        log_fn(f"  Trial 阶段结束, 当前最优: {best['tag']} "
               f"(仿真验证分 {best['sim_val_score']:.2f}, device={best.get('device')})")
        return None

    @stage("seeds")
    def s_seeds():
        best = _best_trial()
        log_fn(f"  最优配置: {best['tag']} ({best['sim_val_score']:.2f}), 开始 3 种子复训")
        cfg = best["config"]
        for seed in (1, 2, 3):
            ckpt = os.path.join(H.MODEL_DIR, f"h1_seed{seed}.pth")
            rec_p = os.path.join(H.RESULT_DIR, "trials", f"seed{seed}.json")
            if os.path.exists(ckpt) and os.path.exists(rec_p):
                continue
            ok = run_trial_worker(cfg, seed, ckpt, rec_p, log_fn)
            if not ok:
                raise RuntimeError(
                    f"种子复训 seed={seed} 双设备尝试后均失败, 阶段中止")
        log_fn("  3 种子复训完成 (方差在 eval 阶段统计)")
        return None

    @stage("finetune")
    def s_finetune():
        records = load_real_records()
        n_by = {s: sum(1 for r in records if r["split"] == s)
                for s in ("Train9", "Val3", "Test3")}
        log_fn(f"  {REAL_DIR_README} | 实际加载: {n_by}")
        assert n_by["Train9"] == 9 and n_by["Val3"] == 3 and n_by["Test3"] == 3, \
            f"真实数据划分异常: {n_by}"

        best = _best_trial()
        ck = torch.load(os.path.join(H.MODEL_DIR, f"h1_{best['tag']}.pth"),
                        map_location="cpu", weights_only=True)
        init_state = ck["model_state_dict"]
        max_ep = 60 if SMOKE else H.REAL_EPOCHS_MAX

        ft_summary = []
        for seed in (1, 2, 3):
            out_p = os.path.join(H.MODEL_DIR, f"h2_ft_seed{seed}.pth")
            rec_p = os.path.join(H.RESULT_DIR, "finetune", f"seed{seed}.json")
            if os.path.exists(out_p) and os.path.exists(rec_p):
                with open(rec_p, encoding="utf-8") as f:
                    ft_summary.append(json.load(f))
                continue
            log_fn(f"  -> 微调 seed={seed} (CPU, Train9 训练 / Val3 早停)")
            state, vscore = finetune_real(
                best["config"], init_state, records, DEVICE, log_fn, seed=seed,
                max_epochs=max_ep, eval_every=H.REAL_EARLY_STOP_EVERY,
                patience=H.REAL_EARLY_STOP_PATIENCE)
            torch.save({"model_state_dict": state, "config": best["config"],
                        "ft_seed": seed, "val3_score": vscore}, out_p)
            os.makedirs(os.path.dirname(rec_p), exist_ok=True)
            with open(rec_p, "w", encoding="utf-8") as f:
                json.dump({"seed": seed, "val3_score": vscore}, f,
                          ensure_ascii=False, indent=1)
            ft_summary.append({"seed": seed, "val3_score": vscore})
        log_fn("  微调完成: Val3 分数 "
               f"{[(d['seed'], round(d['val3_score'], 2)) for d in ft_summary]}")
        return None

    @stage("eval")
    def s_eval():
        import pandas as pd
        cache = np.load(os.path.join(H.RESULT_DIR, "sim_cache.npz"))
        with open(os.path.join(H.RESULT_DIR, "classical_params.json"),
                  encoding="utf-8") as f:
            cp = json.load(f)
        lam_direct = cp.get("nnls_lambda_direct", cp["nnls_lambda"])
        log_fn(f"  冻结参数: 直通NNLS λ={lam_direct}; SG+NNLS λ={cp['nnls_lambda']} "
               f"+ SG(win={cp['sg_win']},poly={cp['sg_poly']}) | 评估设备: CPU")

        # ---- 实际整段 SNR 口径统计(整改: σ 原按首回波归一幅度定义, 易误读) ----
        def seg_snr_db(x, y):
            clean = H.KERNEL_AW @ y
            noise = x - clean
            return float(10.0 * np.log10(np.mean(clean ** 2)
                                         / max(np.mean(noise ** 2), 1e-12)))

        snr_main = [seg_snr_db(x, y) for x, y in zip(cache["tx"][:256], cache["ty"][:256])]
        snr_low = [seg_snr_db(x, y) for x, y in zip(cache["lx"], cache["ly"])]
        snr_note = (f"噪声σ按首回波归一幅度定义; 折算为整段功率SNR: 主测试均值 "
                    f"{np.mean(snr_main):.1f} dB, 压力集均值 {np.mean(snr_low):.1f} dB "
                    f"(范围 {np.min(snr_low):.1f}~{np.max(snr_low):.1f} dB)")

        best = _best_trial()
        sim_ckpt_path = os.path.join(H.MODEL_DIR, f"h1_{best['tag']}.pth")
        sim_ckpt = torch.load(sim_ckpt_path, map_location="cpu", weights_only=True)
        sim_model = build_model(sim_ckpt["config"])
        sim_model.load_state_dict(sim_ckpt["model_state_dict"])

        import hashlib

        def sha256_of(path, _cache={}):
            if path not in _cache:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for blk in iter(lambda: f.read(1 << 20), b""):
                        h.update(blk)
                _cache[path] = h.hexdigest()[:16]
            return _cache[path]

        # 种子方差(独立评估意见: 显式记录各 checkpoint 身份与验证分)
        seed_scores = []
        trials_dir = os.path.join(H.RESULT_DIR, "trials")
        for fn in sorted(os.listdir(trials_dir)):
            if fn.startswith("seed") and fn.endswith(".json") and "cfg" not in fn:
                with open(os.path.join(trials_dir, fn), encoding="utf-8") as f:
                    seed_scores.append(round(json.load(f)["sim_val_score"], 2))
        ident_lines = [
            f"- 协议A被评估权重: `{os.path.basename(sim_ckpt_path)}` "
            f"(sha256前16位 {sha256_of(sim_ckpt_path)}, 记录验证分 "
            f"{best['sim_val_score']:.2f}, device={best.get('device')}, "
            f"steps={best.get('steps')})",
            f"- T06 3种子复训验证分: {seed_scores} "
            f"(极差 {max(seed_scores) - min(seed_scores):.2f})" if seed_scores else "",
        ]

        ft_rec, ft_model = None, None
        ft_dir = os.path.join(H.RESULT_DIR, "finetune")
        if os.path.isdir(ft_dir):
            for fn in os.listdir(ft_dir):
                if not fn.endswith(".json"):
                    continue
                with open(os.path.join(ft_dir, fn), encoding="utf-8") as f:
                    d = json.load(f)
                if ft_rec is None or d["val3_score"] > ft_rec["val3_score"]:
                    ft_rec = d
        if ft_rec is not None:
            ft_path = os.path.join(H.MODEL_DIR, f"h2_ft_seed{ft_rec['seed']}.pth")
            fck = torch.load(ft_path, map_location="cpu", weights_only=True)
            ft_model = build_model(fck["config"])
            ft_model.load_state_dict(fck["model_state_dict"])
            ident_lines.append(
                f"- 协议B被评估权重: `{os.path.basename(ft_path)}` "
                f"(sha256前16位 {sha256_of(ft_path)}, Val3 分 {ft_rec['val3_score']:.2f})")

        legacy_model = None
        _BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _LOCAL_LEGACY = os.path.join(_BUNDLE_ROOT, "checkpoints", "m6_conformer_best_model.pth")
        LEGACY = os.environ.get("LFNMR_LEGACY_CKPT", _LOCAL_LEGACY)
        try:
            try:
                lck = torch.load(LEGACY, map_location="cpu", weights_only=True)
            except Exception:
                # 旧 checkpoint 含 numpy 标量, 严格模式拒载; 本机自产权重, 回退加载
                lck = torch.load(LEGACY, map_location="cpu", weights_only=False)
            lcfg = lck.get("config", {})
            _lm = ConformerResUNet1D(
                stem_dim=lcfg.get("stem_dim", 32),
                d_model=lcfg.get("d_model", 128),
                num_heads=lcfg.get("num_heads", 4),
                num_conformer_layers=lcfg.get("num_layers", 3),
            )
            _lm.load_state_dict(lck["model_state_dict"])   # 成功后才对外可见
            legacy_model = _lm
            log_fn("  已只读加载旧 m6 权重作对照")
        except Exception as e:
            log_fn(f"  旧 m6 对照加载失败(跳过该分支): {e}")

        models = {"Conformer仿真(协议A)": sim_model}
        if ft_model is not None:
            models["Conformer微调(协议B)"] = ft_model
        if legacy_model is not None:
            models["旧m6(数据重用对照)"] = legacy_model
        branches = EV.make_branches(cp["sg_win"], cp["sg_poly"],
                                    cp["nnls_lambda"], models, DEVICE,
                                    nnls_lambda_direct=lam_direct)

        n_eval = 48 if SMOKE else len(cache["tx"])
        log_fn(f"  -> 仿真测试集盲测 ({n_eval} 样本, 绝对真值)")
        t_rows, t_preds = EV.run_branches(
            list(cache["tx"][:n_eval]), list(cache["ty"][:n_eval]),
            branches, DEVICE)
        pd.DataFrame(t_rows).to_csv(
            os.path.join(H.RESULT_DIR, "sim_test_rows.csv"),
            index=False, encoding="utf-8-sig")

        n_low = 24 if SMOKE else len(cache["lx"])
        log_fn("  -> 低信噪比(SNR<15dB)压力子集")
        l_rows, _ = EV.run_branches(
            list(cache["lx"][:n_low]), list(cache["ly"][:n_low]), branches, DEVICE)
        pd.DataFrame(l_rows).to_csv(
            os.path.join(H.RESULT_DIR, "sim_lowsnr_rows.csv"),
            index=False, encoding="utf-8-sig")

        log_fn("  -> 真实 15 记录盲测 (厂商 .Inv 参考; Test3 为封存盲测)")
        records = load_real_records()
        echoes = [r["echo"] for r in records]
        refs = [r["spec"] for r in records]
        r_rows, r_preds = EV.run_branches(
            echoes, refs, branches, DEVICE,
            masks=[r_['valid_mask'] for r_ in records])
        name_map = {i: f"{r['name']}|{r['split']}" for i, r in enumerate(records)}
        for row in r_rows:
            nm = name_map[row["sample"]]
            row["sample_name"], row["split"] = nm.split("|")
        pd.DataFrame(r_rows).to_csv(
            os.path.join(H.RESULT_DIR, "real15_rows.csv"),
            index=False, encoding="utf-8-sig")

        br_names = list(branches.keys())
        EV.render_case_figure(
            cache["tx"][0], cache["ty"][0],
            {n: t_preds[n][0] for n in br_names}, None,
            os.path.join(H.FIGURE_DIR, "case_sim_test0.png"),
            "仿真测试样本 #0 (真值对照)")
        ti = next(i for i, r in enumerate(records) if r["split"] == "Test3")
        EV.render_case_figure(
            echoes[ti], refs[ti], {n: r_preds[n][ti] for n in br_names}, None,
            os.path.join(H.FIGURE_DIR, f"case_real_{records[ti]['name']}.png"),
            f"真实盲测 {records[ti]['name']} (厂商参考, 非真值)")

        dft = pd.DataFrame(t_rows).groupby("branch")[EV.METRIC_KEYS].mean()
        dfl = pd.DataFrame(l_rows).groupby("branch")[EV.METRIC_KEYS].mean()
        dfr = pd.DataFrame(r_rows)
        log_fn("\n  [仿真主测试集(真值) 分支汇总]\n" + dft.round(4).to_string())
        log_fn("\n  [压力集 分支汇总]\n" + dfl.round(4).to_string())
        g2 = dfr.groupby(["branch", "split"])[["pearson_r", "peak_err_log"]].mean()
        log_fn("\n  [真实记录 分支×划分 汇总(r/峰位)]\n" + g2.round(4).to_string())

        summary_lines = [
            f"- 生成时间: {datetime.datetime.now():%Y-%m-%d %H:%M} (v2 修正版)",
            f"- GPU 子进程预算 {GPU_STEPS} 步/trial, CPU 回退预算 {CPU_STEPS} 步 "
            f"(各 trial 实际 device/steps 见 results/trials/*.json)",
            f"- 最优仿真配置: {best['tag']} (验证分 {best['sim_val_score']:.2f})",
            f"- 冻结传统参数: 直通NNLS λ={lam_direct}; SG+NNLS 联合最优 "
            f"λ={cp['nnls_lambda']}, SG(win={cp['sg_win']}, poly={cp['sg_poly']}) "
            f"(在全验证集 1024 样本上选定后冻结, λ 网格扩至 0.5~1000)",
            f"- {snr_note}",
            f"- 协议: {REAL_DIR_README}; 真实记录参考为厂商 .Inv, 非无噪真值",
        ] + ident_lines + [
            "",
            "v2 修正记录(依据 2026-10-24 独立评估): 1) 组分面积函数原误用 "
            "exp(log10网格) 致两列面积误差恒 0, 已改用 T2 真值网格重算; "
            "2) 主测试(1024)与压力集(192)原被拼接成单一'仿真测试'表, 已分表; "
            "3) 传统参数改为全验证集联合选择后重新冻结; 4) 增加 checkpoint "
            "身份(路径/sha256/验证分)与种子方差记录。",
            "",
            "结论按数字如实陈述: 若 Conformer 在某条件下不及 NNLS/SG, 报告如实保留。",
        ]
        EV.write_report(
            {"仿真主测试集(1024, 绝对真值)": t_rows,
             "低幅比压力集(192, 标称σ/首回波 8~15dB 档)": l_rows},
            r_rows, summary_lines,
            os.path.join(H.ROOT_DIR, "最终报告.md"))
        log_fn(f"  报告已写入 {os.path.join(H.ROOT_DIR, '最终报告.md')}")
        return None

    for fn in (s_simdata, s_params, s_trials, s_seeds, s_finetune, s_eval):
        fn()

    log_fn("=" * 72)
    log_fn("全部阶段执行完毕。结果见 results/ 与 最终报告.md; 权重见 models/。")
    log_fn("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        with open(H.MAIN_LOG, "a", encoding="utf-8") as f:
            f.write(traceback.format_exc() + "\n")
        raise
