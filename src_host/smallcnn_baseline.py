# -*- coding: utf-8 -*-
"""smallcnn_baseline.py —— 同预算小型 1D-CNN 对照(隔离架构变量)

目的: 检验 Conformer 架构是否值得使用——对照一个无注意力、无多尺度UNet的
普通残差 CNN, 在完全相同的训练数据(扩展分布流式, 同种子)、同预算(8000步×
batch32)、同损失(UnifiedInversionLoss)、同验证/测试协议下训练与评价。

背景: LMsNN(2025) 为 TensorFlow 实现、无许可证文件、数据格式未文档化,
公平迁移成本高且存在权利歧义; 按审阅建议采用"同预算同协议紧凑CNN"对照,
LMsNN 作为相关工作引用。

架构: 纯卷积骨干(无注意力/无跳跃UNet) + 与 Conformer 相同的池化-MLP谱头。
参数量约为 Conformer 的一半(诚实报告)。
"""
import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import hconfig as H
from simulator import ExtendedStreamDataset
from models import ResBlock1D, UnifiedInversionLoss
from train import eval_model_on_set


class SmallCNN1D(nn.Module):
    """普通 1D 残差 CNN(单尺度, 无注意力), 谱头与 Conformer 相同。"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU())
        self.block1 = nn.Sequential(
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.GELU())
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.GELU())
        self.resblocks = nn.Sequential(*[ResBlock1D(128, 128) for _ in range(4)])
        self.head_pool = nn.AdaptiveAvgPool1d(128)
        self.head_proj = nn.Sequential(nn.Linear(128, 32), nn.GELU(),
                                       nn.Linear(32, 1))

    def forward(self, x):
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.resblocks(h)
        p = self.head_pool(h).transpose(1, 2)
        import torch.nn.functional as F
        return F.softplus(self.head_proj(p).squeeze(-1))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    c = np.load(os.path.join(H.RESULT_DIR, "final_cache.npz"))
    device = torch.device("cpu")

    n_params = sum(p.numel() for p in SmallCNN1D().parameters())
    log(f"SmallCNN1D 参数量: {n_params/1e6:.2f}M (Conformer 为 1.43M)")

    torch.manual_seed(1)
    model = SmallCNN1D().to(device)
    criterion = UnifiedInversionLoss(lambda_huber=1.2, lambda_wass=0.7,
                                     lambda_phys=0.5, lambda_sparse=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=8000, eta_min=1e-5)

    stream = ExtendedStreamDataset(H.STREAM_SEED + 1000 * 1, 32)  # 与A定稿seed1同流
    best_score, best_state = -1e18, None
    t0 = time.time()
    for step in range(1, 8001):
        model.train()
        bx, by, bmask = stream.batch(step)
        bx_t = torch.tensor(bx, device=device).unsqueeze(1)
        by_t = torch.tensor(by, device=device)
        mask_t = torch.tensor(bmask, device=device)
        optimizer.zero_grad()
        pred = model(bx_t)
        loss, _ = criterion(pred, by_t, bx_t, mask_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        if step % 1500 == 0 or step == 8000:
            ev = eval_model_on_set(model, c["vx"], c["vy"], device, masks=c["vm"])
            log(f"step {step}: 扩展验证 r={ev['mean_r']:.4f} "
                f"峰={ev['mean_peak_err']:.4f} 分={ev['score']:.2f}")
            if ev["score"] > best_score:
                best_score = ev["score"]
                best_state = copy.deepcopy(model.state_dict())
    log(f"训练完成: 最佳扩展验证分 {best_score:.2f}, 用时 {time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    out = os.path.join(H.MODEL_DIR, "h1_smallcnn_seed1.pth")
    torch.save({"model_state_dict": best_state, "arch": "SmallCNN1D",
                "params": n_params, "ext_val_score": best_score,
                "budget": "8000步×batch32, 同流种子1042"}, out)
    log(f"已保存: {out}")

    # 测试集评价(含 mask 评分, 与修正基线同口径)
    tx, ty, tm = c["tx"], c["ty"], c["tm"]
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tx), 64):
            preds.append(model(torch.tensor(tx[i:i+64]).unsqueeze(1)).numpy())
    preds = np.concatenate(preds)
    from baselines import spectrum_metrics
    trunc = tm.sum(1) < 1024
    rows = []
    for i in range(len(tx)):
        m = spectrum_metrics(preds[i], ty[i], tx[i], tm[i])
        m["truncated"] = bool(trunc[i])
        rows.append(m)
    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby("truncated")[["pearson_r", "peak_err_log", "spec_l2"]].mean().round(4)
    print("\nSmallCNN 测试集(修正评分口径):")
    print(g.to_string())
    df.to_csv(os.path.join(H.ROOT_DIR, "13_现代基线对照_20261025",
                           "smallcnn_test_rows.csv"),
              index=False, encoding="utf-8-sig")
    log("测试行已保存")


if __name__ == "__main__":
    main()
