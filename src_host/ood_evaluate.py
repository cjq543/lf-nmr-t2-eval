# -*- coding: utf-8 -*-
"""
ood_evaluate.py —— 分布外(OOD)鲁棒性评价(独立评估建议的"训练范围外测试")

在【训练分布之外】的 7 个维度上, 用固定冻结权重评价 4 个分支:
  直通NNLS / SG+NNLS(冻结参数) / A仿真版T06 / B微调版seed1

OOD 维度(均为训练生成器未覆盖或边缘的范围):
  V1 更窄峰宽   σ∈[0.05,0.12]        (训练主峰 σ∈[0.15,0.40])
  V2 更宽峰宽   σ∈[0.45,0.70]
  V3 峰位外推   主峰 T2∈[30,80]ms 与 [900,2500]ms (训练主峰 [80,900])
  V4 多组分     4~5 峰 (训练 1~4 峰)
  V5 强基线漂移 直流/漂移幅度放大10倍
  V6 缺失时间窗 包络 600ms / 1200ms 后置零(模拟短记录; 前向指标不适用, 跳过)
  V7 极低幅比   σ/首回波 ∈ [5,8]dB 档 (训练压力档最低 8dB)

用法: python src/ood_evaluate.py   (只依赖本包文件与缓存, 无需真实数据)
输出: results/ood_rows.csv + OOD评估报告.md(自动汇总各维度胜负)
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import hconfig as H
from hconfig import STD_LOG_T2, KERNEL_AW
from baselines import nnls_invert, sg_nnls_invert, spectrum_metrics
from train import build_model, predict_spectra

N_PER_VARIANT = 128
BASE_SEED = 707000   # 与训练流/验证/主测试种子均不相交

VARIANTS = [
    "V1_更窄峰宽", "V2_更宽峰宽",
    "V3_峰位外推_快端", "V3_峰位外推_慢端",
    "V4_多组分4-5峰", "V5_强基线漂移",
    "V6_缺失时间窗_600ms", "V6_缺失时间窗_1200ms",
    "V7_极低幅比5-8dB",
]


def _add_peak(spec, mu, sig, amp):
    spec += amp * np.exp(-0.5 * ((STD_LOG_T2 - mu) / sig) ** 2)


def sample_variant(rng, variant):
    """按变体生成 (输入包络, 真值谱)。归一化约定与训练完全一致。"""
    # --- 谱 ---
    mus, sigmas, amps = [], [], []
    if variant == "V1_更窄峰宽":
        mus = [np.log10(rng.uniform(80, 900))]
        sigmas = [rng.uniform(0.05, 0.12)]
        amps = [rng.uniform(0.5, 1.0)]
    elif variant == "V2_更宽峰宽":
        mus = [np.log10(rng.uniform(80, 900))]
        sigmas = [rng.uniform(0.45, 0.70)]
        amps = [rng.uniform(0.5, 1.0)]
    elif variant == "V3_峰位外推_快端":
        mus = [np.log10(rng.uniform(30, 80))]
        sigmas = [rng.uniform(0.15, 0.40)]
        amps = [rng.uniform(0.5, 1.0)]
    elif variant == "V3_峰位外推_慢端":
        mus = [np.log10(rng.uniform(900, 2500))]
        sigmas = [rng.uniform(0.15, 0.40)]
        amps = [rng.uniform(0.5, 1.0)]
    elif variant == "V4_多组分4-5峰":
        k = int(rng.integers(4, 6))
        mus = sorted(rng.uniform(np.log10(5.0), np.log10(2500.0), size=k))
        sigmas = [rng.uniform(0.10, 0.25) for _ in range(k)]
        amps = [rng.uniform(0.1, 1.0) for _ in range(k)]
    else:  # V5/V6/V7 用标准谱型(与训练同分布), 只改噪声/时间窗
        mu_main = np.log10(rng.uniform(80, 900))
        sig_main = rng.uniform(0.15, 0.40)
        amp_main = rng.uniform(0.5, 1.0)
        mus, sigmas, amps = [mu_main], [sig_main], [amp_main]
        if rng.random() < 0.6:
            mus.append(np.log10(rng.uniform(10, 80)))
            sigmas.append(rng.uniform(0.12, 0.35))
            amps.append(rng.uniform(0.1, 0.6) * amp_main)
        if rng.random() < 0.3:
            mus.append(np.log10(rng.uniform(500, 2000)))
            sigmas.append(rng.uniform(0.10, 0.25))
            amps.append(rng.uniform(0.03, 0.10) * amp_main)

    spec = np.zeros_like(STD_LOG_T2, dtype=np.float64)
    for mu, sig, amp in zip(mus, sigmas, amps):
        _add_peak(spec, mu, sig, amp)
    y = KERNEL_AW @ spec
    scale = float(y[0]) if y[0] > 1e-9 else 1.0

    # --- 噪声(与训练同型, 按变体覆盖幅度) ---
    if variant == "V7_极低幅比5-8dB":
        snr = rng.uniform(5.0, 8.0)
    elif rng.random() < 0.10:
        snr = rng.uniform(8.0, 15.0)
    else:
        snr = np.exp(rng.uniform(np.log(15.0), np.log(45.0)))
    sigma = 10.0 ** (-snr / 20.0)
    n = y.size
    if variant == "V5_强基线漂移":
        noise = (rng.normal(0, sigma, n) + rng.normal(0, 0.04)
                 + rng.normal(0, 0.03) * (np.arange(n) / n))
    else:
        noise = (rng.normal(0, sigma, n) + rng.normal(0, 0.004)
                 + rng.normal(0, 0.002) * (np.arange(n) / n))
    x = y / scale + noise

    # --- 缺失时间窗(仅截断输入, 真值谱保持完整) ---
    if variant.startswith("V6"):
        t_cut = 600.0 if "600" in variant else 1200.0
        cut = int(np.searchsorted(H.STD_ECHO_TIMES, t_cut))
        x[cut:] = 0.0

    return x.astype(np.float32), (spec / scale).astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", default="h1_T06_退火长训.pth",
                    help="models/ 下的 A 模型文件名")
    ap.add_argument("--model-b", default="h2_ft_seed1.pth",
                    help="models/ 下的 B 模型文件名, 传 none 跳过 B 分支")
    ap.add_argument("--params", default="classical_params.json",
                    help="results/ 下的冻结参数文件名")
    ap.add_argument("--out-suffix", default="",
                    help="输出文件名后缀, 如 _final")
    args = ap.parse_args()

    with open(os.path.join(H.RESULT_DIR, args.params), encoding="utf-8") as f:
        cp = json.load(f)
    lam_dir = cp.get("nnls_lambda_direct", cp["nnls_lambda"])

    ckA = torch.load(os.path.join(H.MODEL_DIR, args.model_a),
                     map_location="cpu", weights_only=True)
    mA = build_model(ckA["config"]); mA.load_state_dict(ckA["model_state_dict"])
    mB = None
    if args.model_b.lower() != "none":
        ckB = torch.load(os.path.join(H.MODEL_DIR, args.model_b),
                         map_location="cpu", weights_only=True)
        mB = build_model(ckB["config"]); mB.load_state_dict(ckB["model_state_dict"])

    rows = []
    t0 = time.time()
    for variant in VARIANTS:
        xs, ys = [], []
        for k in range(N_PER_VARIANT):
            rng = np.random.default_rng(BASE_SEED + hash_local(variant) + k)
            x, y = sample_variant(rng, variant)
            xs.append(x); ys.append(y)
        preds_A = predict_spectra(mA, xs, torch.device("cpu"))
        preds_B = (predict_spectra(mB, xs, torch.device("cpu"))
                   if mB is not None else None)
        # P1修正: 截断样本的有效观测窗直接传入求解器
        from hconfig import STD_ECHO_TIMES
        vs = None
        if variant.startswith("V6"):
            t_cut = 600.0 if "600" in variant else 1200.0
            vs = [(STD_ECHO_TIMES <= t_cut).astype(np.float32)] * N_PER_VARIANT

        for i in range(N_PER_VARIANT):
            echo_arg = None if variant.startswith("V6") else xs[i]
            v_mask = vs[i] if vs is not None else None
            pairs = [
                ("直通NNLS", nnls_invert(xs[i], lam_dir, v_mask)),
                (f"SG({cp['sg_win']},{cp['sg_poly']})+NNLS",
                 sg_nnls_invert(xs[i], cp["sg_win"], cp["sg_poly"],
                                cp["nnls_lambda"], v_mask)),
                ("A仿真版", preds_A[i]),
            ]
            if preds_B is not None:
                pairs.append(("B微调版", preds_B[i]))
            for branch, pred in pairs:
                m = spectrum_metrics(pred, ys[i], echo_arg)
                if echo_arg is None:
                    m.pop("fwd_rmse", None)
                m["variant"] = variant; m["branch"] = branch
                rows.append(m)
        print(f"  [{variant}] 完成 (累计 {time.time()-t0:.0f}s)", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(H.RESULT_DIR, f"ood_rows{args.out_suffix}.csv"),
              index=False, encoding="utf-8-sig")

    keys = ["pearson_r", "peak_err_log", "spec_l2", "area_short_err"]
    piv = df.groupby(["variant", "branch"])[keys].mean().round(4)

    lines = ["# 分布外(OOD)鲁棒性评价报告(自动生成)", "",
             f"- 每维度 {N_PER_VARIANT} 样本, 种种子 {BASE_SEED}(与训练/验证/测试不相交)",
             f"- 分支: 直通NNLS(λ={lam_dir}) / SG+NNLS(冻结联合参数) / A仿真版 / B微调版",
             "- V6 缺失时间窗不计算前向RMSE; 其余指标与主报告同口径", "",
             piv.to_markdown(), "",
             "## 各维度平均谱相关 r 一览", ""]
    wide = df.pivot_table(index="variant", columns="branch", values="pearson_r")
    wide = wide.round(4)
    lines.append(wide.to_markdown())
    lines += ["", "## 自动结论", ""]
    for variant in VARIANTS:
        sub = df[df["variant"] == variant].groupby("branch")["pearson_r"].mean()
        best, worst = sub.idxmax(), sub.idxmin()
        a_r = sub.get("A仿真版", np.nan)
        sg_r = sub.get(f"SG({cp['sg_win']},{cp['sg_poly']})+NNLS", np.nan)
        rel = "优于" if a_r > sg_r else "不及"
        lines.append(f"- {variant}: 最佳={best}({sub.max():.4f}); A({a_r:.4f}) "
                     f"{rel} SG+NNLS({sg_r:.4f})")
    out_md = os.path.join(H.ROOT_DIR, f"OOD评估报告{args.out_suffix}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已写入 {out_md}")
    print("\n[各维度 平均r]")
    print(wide.to_string())


def hash_local(name: str) -> int:
    """稳定字符串->整数(不用内置 hash, 其跨进程不稳定)"""
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) % 1000003
    return h


if __name__ == "__main__":
    main()
