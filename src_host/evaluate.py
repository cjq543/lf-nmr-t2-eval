# -*- coding: utf-8 -*-
"""
evaluate.py —— 冻结参数后的统一盲测与报告生成

原则:
  * 仿真测试集(种子与训练流/验证集完全不相交): 有真值, 四分支对照;
  * 真实 15 记录: 厂商 .Inv 作参考(明确标注非真值), Train/Val 汇总与 Test3
    盲测分行报告; 顺带只读加载旧 m6 权重作"数据重用旧模型"对照;
  * 所有分支吃完全相同的输入(单位化包络), 仅去噪/反演方法不同。
"""

import os
import numpy as np
import torch

from hconfig import RESULT_DIR, FIGURE_DIR, STD_T2_MS, STD_LOG_T2, trapz_log
from baselines import nnls_invert, sg_nnls_invert, spectrum_metrics
from train import predict_spectra

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METRIC_KEYS = ["pearson_r", "peak_err_log", "spec_l2",
               "area_short_err", "area_long_err", "fwd_rmse"]


def _agg(rows):
    out = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
        out[k + "_std"] = float(np.std(vals)) if vals else float("nan")
    out["n"] = len(rows)
    return out


def run_branches(echoes, refs, branches, device, masks=None):
    """branches: {name: callable(echoes, masks) -> (N,128) 谱}。返回逐样本行表。
    masks 提供时(截断样本), 同时传递给传统求解器(有效观测窗)和评分(前向RMSE)。"""
    rows = []
    preds_all = {name: fn(echoes, masks) for name, fn in branches.items()}
    for i in range(len(echoes)):
        for name, preds in preds_all.items():
            m = spectrum_metrics(preds[i], refs[i], echoes[i],
                                 None if masks is None else masks[i])
            m["branch"] = name
            m["sample"] = i
            rows.append(m)
    return rows, preds_all


def make_branches(sg_win, sg_poly, nnls_lambda_sg, models, device,
                  nnls_lambda_direct=None):
    """models: {name: nn.Module}。直通与 SG 分支可使用各自冻结的 λ。
    2026-10-25 P1修正: 传统分支签名改为 (echoes, masks), mask 直接传入求解器。"""
    if nnls_lambda_direct is None:
        nnls_lambda_direct = nnls_lambda_sg

    def _nnls_branch(es, ms):
        if ms is None:
            return np.stack([nnls_invert(e, nnls_lambda_direct) for e in es])
        return np.stack([nnls_invert(e, nnls_lambda_direct, m)
                         for e, m in zip(es, ms)])

    def _sg_branch(es, ms):
        if ms is None:
            return np.stack([sg_nnls_invert(e, sg_win, sg_poly, nnls_lambda_sg)
                             for e in es])
        return np.stack([sg_nnls_invert(e, sg_win, sg_poly, nnls_lambda_sg, m)
                         for e, m in zip(es, ms)])

    branches = {
        "直通NNLS": _nnls_branch,
        f"SG({sg_win},{sg_poly})+NNLS": _sg_branch,
    }
    for name, mdl in models.items():
        mdl = mdl.to(device).eval()
        branches[name] = lambda es, ms, m=mdl: predict_spectra(m, es, device)
    return branches


def render_case_figure(echoes, refs, preds_all, meta, out_png, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    seg = slice(0, 220)
    ax.plot(echoes[seg], color="#999999", lw=0.8, label="输入包络(单位化)")
    ax.set_title("输入回波包络(前 220 点)")
    ax.set_xlabel("回波序号(1024 点网格)")
    ax.set_ylabel("归一化幅度")
    ax.legend(fontsize=8); ax.grid(ls="--", alpha=0.4)

    ax = axes[1]
    ax.plot(STD_T2_MS, refs, "k-", lw=2.4, label="参考谱")
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd"]
    for j, (name, preds) in enumerate(preds_all.items()):
        ax.plot(STD_T2_MS, preds, ls="--", lw=1.4, color=colors[j % len(colors)],
                label=name)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 5000.0)
    ax.set_xlabel("T2 (ms)")
    ax.set_ylabel("谱密度(log 面积归一)")
    ax.set_title(title)
    ax.legend(fontsize=7.5); ax.grid(ls="--", alpha=0.4, which="both")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_report(sim_tables, real_rows, summary_lines, out_md):
    """sim_tables: {表名: 逐样本行列表}。整改: 主测试与压力集必须分表,
    不得拼接混入同一"仿真测试"口径(2026-10-24 独立评估意见)。"""
    import pandas as pd
    lines = ["# 上位机大模型 v2 训练与盲测报告(自动生成)", ""]
    lines += summary_lines
    sec = 1
    for table_name, rows in sim_tables.items():
        df = pd.DataFrame(rows)
        lines += ["", f"## {sec}. {table_name}(n={len(df)})分分支汇总", ""]
        if len(df):
            g = df.groupby("branch")[METRIC_KEYS].mean().round(5)
            lines.append(g.to_markdown())
        sec += 1
    lines += ["", f"## {sec}. 真实 15 记录(厂商 .Inv 参考, 非真值)逐记录明细", ""]
    dfr = pd.DataFrame(real_rows)
    if len(dfr):
        cols = ["sample_name", "split", "branch"] + METRIC_KEYS
        lines.append(dfr[cols].round(5).to_markdown(index=False))
        lines += ["", "### 分支 × 数据划分汇总", ""]
        g2 = dfr.groupby(["branch", "split"])[METRIC_KEYS].mean().round(5)
        lines.append(g2.to_markdown())
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
