# -*- coding: utf-8 -*-
"""
cascade_experiment.py —— 上下位机级联对照实验(笔记本侧, 需真实数据)

问题: 上位机反演模型 A 此前的真实记录输入是厂商 .pea 包络(厂商已提取)。
本实验固定上位机 A(与 NNLS 基线), 比较四种上游输入路径的端到端反演效果:

  P1 厂商.pea包络直连      (此前 v2 评价的输入, 作为参照)
  P2 原始Fid + 自研锚点提取  (100k FIR 对齐到 50k, 相位投影, 锚点 t_pea-0.24ms)
  P3 SG(21,2)平滑 + 同上    (传统去噪在链路中的贡献)
  P4 下位机小模型去噪 + 同上 (板端 ECA-DS-TCN 残差模型, 与部署形态一致)

每种输入分别用 A(h1_T06) 与 NNLS(旧冻结参数) 反演, 以厂商 .Inv 为参考
(非真值) 报告 r / 主峰误差。评价 15 条真实记录, Test3 单列。

数据与旧代码全部只读; 复用 portable_nmr_eval_v4 的 legacy_io/timing_alignment/
model_adapter(下位机模型独立推理适配器)。
用法: python src/cascade_experiment.py
"""
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
from hconfig import STD_ECHO_TIMES
from realdata import load_real_records
from train import build_model, predict_spectra
from baselines import nnls_invert, spectrum_metrics

# Frozen on-board residual denoiser checkpoint (SHA-256 verified in MANIFEST_SHA256.csv)
_LOCAL_EDGE = os.path.join(_BUNDLE_ROOT, "checkpoints", "m4_cpmg_multiecho_residual_eca_ds_tcn.pth")
EDGE_MODEL_PATH = os.environ.get("LFNMR_EDGE_CKPT", _LOCAL_EDGE)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---- 复用 v4 包: IO 与 50kHz 对齐 ----
from legacy_io import load_fid, load_pea, read_par  # noqa: E402
from timing_alignment import align_to_50k, candidate_mask  # noqa: E402
from model_adapter import load_predictor  # noqa: E402


def apply_edge_model_streaming(predict_fn, iq_50k, times_50k, te_ms,
                               win_size=1024, hop_size=512):
    """下位机模型流式应用(与 m7 部署形态一致: 反射填充+汉宁拼接+候选掩码)。"""
    n_pts = iq_50k.shape[1]
    pad_len = win_size // 2
    pad_iq = np.pad(iq_50k, ((0, 0), (pad_len, pad_len)), mode="reflect")
    pad_times = np.pad(times_50k, (pad_len, pad_len), mode="edge")

    accum_iq = np.zeros_like(pad_iq)
    accum_w = np.zeros(pad_iq.shape[1])
    hann = np.hanning(win_size)

    starts = list(range(0, pad_iq.shape[1] - win_size + 1, hop_size))
    chunks, masks, scales = [], [], []
    for st in starts:
        chunk = pad_iq[:, st:st + win_size]
        c_mask = candidate_mask(pad_times[None, st:st + win_size], te_ms)
        sc = max(np.max(np.abs(chunk)), 1e-6)
        chunks.append(chunk / sc)
        masks.append(c_mask)
        scales.append(sc)

    inp = np.stack(chunks).astype(np.float32)
    msk = np.stack(masks).astype(np.float32)
    clean = predict_fn(inp, msk)

    for idx, st in enumerate(starts):
        accum_iq[:, st:st + win_size] += clean[idx] * scales[idx] * hann
        accum_w[st:st + win_size] += hann
    vw = accum_w > 1e-6
    accum_iq[:, vw] /= accum_w[vw]
    return accum_iq[:, pad_len:pad_len + n_pts]


def extract_envelope(iq_50k, times_50k, pea_times_ms):
    """锚点(t_pea-0.24ms)提取回波包络, 前20回波估相位投影实轴。"""
    anchors = np.clip(np.searchsorted(times_50k, pea_times_ms - 0.24),
                      1, iq_50k.shape[1] - 2)
    ei = np.array([np.mean(iq_50k[0, k - 1:k + 2]) for k in anchors])
    eq = np.array([np.mean(iq_50k[1, k - 1:k + 2]) for k in anchors])
    ref = min(20, len(anchors))
    phase = np.arctan2(np.mean(eq[:ref]), np.mean(ei[:ref]))
    return ei * np.cos(phase) + eq * np.sin(phase)


def envelope_to_model_input(env, pea_times):
    """包络 -> 1024 点标准网格输入(截断区补零+mask), 首点归一。"""
    x = np.interp(STD_ECHO_TIMES, pea_times, env)
    valid = (STD_ECHO_TIMES <= pea_times[-1]).astype(np.float32)
    x[valid < 0.5] = 0.0
    s = float(x[0])
    if abs(s) < 1e-9:
        s = 1.0
    return (x / s).astype(np.float32), valid


def main():
    records = load_real_records()
    with open(os.path.join(H.RESULT_DIR, "classical_params.json"),
              encoding="utf-8") as f:
        import json
        cp = json.load(f)
    lam_dir = cp.get("nnls_lambda_direct", cp["nnls_lambda"])

    ck = torch.load(os.path.join(H.MODEL_DIR, "h1_T06_退火长训.pth"),
                    map_location="cpu", weights_only=True)
    modelA = build_model(ck["config"])
    modelA.load_state_dict(ck["model_state_dict"])

    log("加载下位机小模型(ECADSTCNResidual, 只读)...")
    predict_fn, meta = load_predictor(EDGE_MODEL_PATH, device="cpu")

    import pandas as pd
    mdf = pd.read_csv(H.SPLIT_MANIFEST, encoding="utf-8-sig")
    mdf = mdf[mdf["seq_name"] == "CPMG"].set_index("sample_name")

    rows = []
    t0 = time.time()
    for r in records:
        row = mdf.loc[r["name"]]
        te_ms = float(row["te_ms"])
        stem = str(row["par_path"])[:-4]

        iq_100k, time_100k, _ = load_fid(stem + ".1.Fid")
        pea_arr, _ = load_pea(stem + ".1.pea")
        pea_times = pea_arr[:, 0]

        iq_50k, time_50k, _ = align_to_50k(iq_100k, time_100k)

        sg_iq = np.zeros_like(iq_50k)
        sg_iq[0] = savgol_filter(iq_50k[0], 21, 2)
        sg_iq[1] = savgol_filter(iq_50k[1], 21, 2)

        log(f"{r['name']:<12}({r['split']}) FIR对齐完成, 下位机模型推理中 "
            f"({iq_50k.shape[1]} 点)")
        edge_iq = apply_edge_model_streaming(predict_fn, iq_50k, time_50k, te_ms)

        paths = {
            "P1厂商pea直连": pea_arr[:, 1],
            "P2原始Fid锚点": extract_envelope(iq_50k, time_50k, pea_times),
            "P3_SG+Fid锚点": extract_envelope(sg_iq, time_50k, pea_times),
            "P4下位机模型+Fid锚点": extract_envelope(edge_iq, time_50k, pea_times),
        }
        for pname, env in paths.items():
            if len(env) != len(pea_times):      # P1 即 pea 长度, 其余同锚点数
                continue
            x, valid = envelope_to_model_input(env, pea_times)
            for iname, spec in (("A反演", predict_spectra(
                    modelA, [x], torch.device("cpu"))[0]),
                    ("NNLS反演", nnls_invert(x, lam_dir, valid))):
                m = spectrum_metrics(spec, r["spec"], x, valid)
                m.pop("fwd_rmse", None)
                m["input_path"] = pname
                m["inverter"] = iname
                m["sample_name"] = r["name"]
                m["split"] = r["split"]
                rows.append(m)
        log(f"  完成, 累计 {time.time()-t0:.0f}s")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(H.RESULT_DIR, "cascade_rows.csv"),
              index=False, encoding="utf-8-sig")
    g = df.groupby(["input_path", "inverter"])[
        ["pearson_r", "peak_err_log"]].mean().round(4)
    print("\n==== 级联对照: 输入路径 × 反演器 (全部15记录均值) ====")
    print(g.to_string())
    gt = df[df["split"] == "Test3"].groupby(["input_path", "inverter"])[
        ["pearson_r", "peak_err_log"]].mean().round(4)
    print("\n==== Test3 留出 3 记录 ====")
    print(gt.to_string())

    lines = ["# 上下位机级联对照实验(自动生成)", "",
             f"- 上位机A = h1_T06(仿真版); NNLS = 旧冻结直通参数 λ={lam_dir}",
             "- 输入路径: P1厂商.pea包络直连 / P2原始Fid+锚点提取 / "
             "P3 SG(21,2)+锚点 / P4 下位机小模型+锚点",
             "- 参考 = 厂商 .Inv(非真值); 50kHz对齐=81抽头FIR+群延迟补偿; "
             "锚点 t_pea-0.24ms, 前20回波相位投影", "",
             "## 全部15记录", "", g.to_markdown(), "",
             "## Test3 留出", "", gt.to_markdown(), "",
             "注: P1 为此前 v2 评价输入; P2~P4 为自研全链路(不依赖厂商包络提取)。"]
    with open(os.path.join(H.ROOT_DIR, "级联实验报告.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("报告已写入 级联实验报告.md")


if __name__ == "__main__":
    main()
