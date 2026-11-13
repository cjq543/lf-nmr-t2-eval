# -*- coding: utf-8 -*-
"""
cascade_lowsnr.py —— 低信噪比上下位机级联实验(纯笔记本, 无需板卡)

补齐级联故事的最后一块: 此前级联实验(高信噪桌面数据)显示上游去噪对锚点
包络反演无实质增益; 本实验在 5.3 节同款噪声阶梯(0.1/0.3/1.0 倍复数RMS)
下重跑同一路径, 检验"低信噪条件下下位机去噪→上位机反演的级联增益"。

设计: 对 15 条真实 Fid, 50kHz 对齐后注入高斯白噪(固定种子, 每(记录,档位)确定),
三分支(原始带噪/SG(21,2)/下位机ECA-DS-TCN) → 锚点提取 → A(h1_T06) 与
NNLS(旧冻结λ) 反演 → 对厂商 .Inv 参考评 r/主峰误差。另含 0 倍(洁净)基准。

输出: results/cascade_lowsnr_rows.csv + 级联低信噪实验报告.md
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4_DIR = os.environ.get("LFNMR_V4_DIR", os.path.join(_BUNDLE_ROOT, "external_tools", "portable_nmr_eval_v4"))
sys.path.insert(0, V4_DIR)

import hconfig as H
from realdata import load_real_records
from train import build_model, predict_spectra
from baselines import nnls_invert, spectrum_metrics

_LOCAL_EDGE = os.path.join(_BUNDLE_ROOT, "checkpoints", "m4_cpmg_multiecho_residual_eca_ds_tcn.pth")
EDGE_MODEL_PATH = os.environ.get("LFNMR_EDGE_CKPT", _LOCAL_EDGE)
NOISE_LEVELS = [0.0, 0.1, 0.3, 1.0]
BASE_SEED = 20261024

from legacy_io import load_fid, load_pea  # noqa: E402
from timing_alignment import align_to_50k, candidate_mask  # noqa: E402
from model_adapter import load_predictor  # noqa: E402
from cascade_experiment import (apply_edge_model_streaming,  # noqa: E402
                                extract_envelope, envelope_to_model_input)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    import pandas as pd
    records = load_real_records()
    mdf = pd.read_csv(H.SPLIT_MANIFEST, encoding="utf-8-sig")
    mdf = mdf[mdf["seq_name"] == "CPMG"].set_index("sample_name")
    with open(os.path.join(H.RESULT_DIR, "classical_params.json"),
              encoding="utf-8") as f:
        cp = json.load(f)
    lam_dir = cp.get("nnls_lambda_direct", cp["nnls_lambda"])

    ck = torch.load(os.path.join(H.MODEL_DIR, "h1_T06_退火长训.pth"),
                    map_location="cpu", weights_only=True)
    modelA = build_model(ck["config"])
    modelA.load_state_dict(ck["model_state_dict"])
    log("加载下位机小模型(只读)...")
    predict_fn, _ = load_predictor(EDGE_MODEL_PATH, device="cpu")

    rows = []
    t0 = time.time()
    NOISE_SEEDS = (0, 1, 2)   # 3种子(整改: 单种子下0.1/0.3档出现反常波动)
    for idx, r in enumerate(records):
        row = mdf.loc[r["name"]]
        te_ms = float(row["te_ms"])
        stem = str(row["par_path"])[:-4]
        iq_100k, time_100k, _ = load_fid(stem + ".1.Fid")
        pea_arr, _ = load_pea(stem + ".1.pea")
        pea_times = pea_arr[:, 0]
        iq_50k, time_50k, _ = align_to_50k(iq_100k, time_100k)
        rms = float(np.sqrt(np.mean(iq_50k[0] ** 2 + iq_50k[1] ** 2)))

        for li, level in enumerate(NOISE_LEVELS):
          for sk in NOISE_SEEDS:
            rng = np.random.default_rng(BASE_SEED + 97 * li + 1000 * sk + idx)
            if level > 0:
                noisy = iq_50k + rng.normal(0.0, level * rms,
                                            size=iq_50k.shape)
            else:
                if sk > 0:
                    continue          # 0倍档与种子无关, 只算一次
                noisy = iq_50k
            sg_iq = np.zeros_like(noisy)
            sg_iq[0] = savgol_filter(noisy[0], 21, 2)
            sg_iq[1] = savgol_filter(noisy[1], 21, 2)
            edge_iq = apply_edge_model_streaming(predict_fn, noisy, time_50k,
                                                 te_ms)
            for pname, iq in (("原始带噪", noisy), ("SG", sg_iq),
                              ("下位机模型", edge_iq)):
                env = extract_envelope(iq, time_50k, pea_times)
                x, valid = envelope_to_model_input(env, pea_times)
                for iname, spec in (
                    ("A反演", predict_spectra(modelA, [x],
                                              torch.device("cpu"))[0]),
                    ("NNLS反演", nnls_invert(x, lam_dir, valid)),
                ):
                    m = spectrum_metrics(spec, r["spec"], x, valid)
                    m.pop("fwd_rmse", None)
                    m["noise_level"] = level
                    m["input_path"] = pname
                    m["inverter"] = iname
                    m["seed_k"] = sk
                    m["sample_name"] = r["name"]
                    m["split"] = r["split"]
                    rows.append(m)
        log(f"{r['name']:<12} 完成 (累计 {time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(H.RESULT_DIR, "cascade_lowsnr_rows.csv"),
              index=False, encoding="utf-8-sig")
    g = df.groupby(["noise_level", "input_path", "inverter"])[
        ["pearson_r", "peak_err_log"]].mean().round(4)
    print("\n" + g.to_string())

    lines = ["# 低信噪比上下位机级联实验(自动生成, v2 三种子版)", "",
             f"- 噪声阶梯: 0/0.1/0.3/1.0 倍复数RMS(与5.3节同标尺); "
             f"噪声档每档 3 个独立种子({BASE_SEED} 系), 表内为种子间平均", "",
             g.to_markdown(), "",
             "## 自动解读", ""]
    for lv in NOISE_LEVELS:
        sub = df[df["noise_level"] == lv].groupby("input_path")["pearson_r"].mean()
        edge_r = sub.get("下位机模型", np.nan)
        raw_r = sub.get("原始带噪", np.nan)
        sg_r = sub.get("SG", np.nan)
        judge = "去噪路径领先" if edge_r > raw_r + 0.005 else (
            "无实质差异" if abs(edge_r - raw_r) <= 0.005 else "去噪路径落后")
        lines.append(f"- 噪声 {lv} 倍RMS: 下位机模型 {edge_r:.4f} vs 原始 {raw_r:.4f}"
                     f" vs SG {sg_r:.4f} -> {judge}")
    with open(os.path.join(H.ROOT_DIR, "级联低信噪实验报告.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("报告已写入 级联低信噪实验报告.md")


if __name__ == "__main__":
    main()
