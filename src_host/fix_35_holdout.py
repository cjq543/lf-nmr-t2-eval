# -*- coding: utf-8 -*-
"""fix_35_holdout.py —— v3：修正真正的 pea 文件读取，冻结权重再分析。
2026-10-26：原 v2 的 pea 分支误读 Fid。本版结果写入 21 号目录，保留原 CSV。
本版为已查看历史记录的协议修正/诊断，不是新的盲测或前瞻性验证。
修正项（依据 19_35条留出结果复核）:
  1. 统一指标: 全部调用 spectrum_metrics（三点内插峰位、面积归一、有效窗前向RMSE）
  2. 补 SG 峰位/L₂ 等结果
  3. 输入协议标注为"原始Fid模值直接插值（协议偏离诊断）"，与既有 pea 包络协议区分
  4. pea 输入诊断结果并列保留（SG旧 0.904 > A定稿 0.893 的排序变化如实报告）
"""
import json
import os
import sys
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import hconfig as H
from hconfig import STD_ECHO_TIMES, STD_LOG_T2
from baselines import (nnls_invert, sg_nnls_invert, pearson_r,
                       spectrum_metrics, peak_pos_log)
from realdata import _parse_columns
from train import build_model, predict_spectra

WORK_ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("LFNMR_HOLDOUT35_DATA_DIR", str(WORK_ROOT / "external_data" / "benchtop_holdout35"))
LOCAL = Path(os.environ.get("LFNMR_NIUMAG_DIR", str(WORK_ROOT / "external_data" / "benchtop_niumag")))
CAMPAIGNS = [(LOCAL / "04_第二次油样品_椰橄芝花", "第二次油样品"),
             (LOCAL / "05_第三次油样品_豆棉玉", "第三次油样品")]


def load_m(pth):
    ck = torch.load(os.path.join(H.MODEL_DIR, pth), map_location="cpu",
                    weights_only=True)
    m = build_model(ck["config"])
    m.load_state_dict(ck["model_state_dict"])
    return m


def load_fid_mag(p):
    a = np.loadtxt(p, comments="#")
    if a.ndim == 1:
        a = a.reshape(1, -1)
    t = a[:, 3] if a.shape[1] > 3 else np.arange(len(a)) * 0.02
    mag = np.abs(a[:, 0] + 1j * a[:, 1])
    return t, mag


def read_inv(p):
    a = []
    for line in open(p, encoding="gbk", errors="ignore"):
        parts = line.split()
        if len(parts) >= 2:
            try:
                a.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
    a = np.array(a)
    return a[:, 0], a[:, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path,
                    default=WORK_ROOT / "21_执行与模型决策_20261026" / "corrected_v3")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cp = json.load(open(os.path.join(
        H.RESULT_DIR, "classical_params_masked_20261025.json"),
        encoding="utf-8"))
    lam = cp.get("nnls_lambda_direct", 1000.0)
    sg_win, sg_poly, sg_lam = cp["sg_win"], cp["sg_poly"], cp["nnls_lambda"]

    records = []
    for camp_root, campaign in CAMPAIGNS:
        for cond_dir in sorted(os.listdir(camp_root)):
            cdir = os.path.join(camp_root, cond_dir)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                if not fn.endswith(".par"):
                    continue
                stem = fn[:-4]
                fid = os.path.join(cdir, stem + ".1.Fid")
                inv = os.path.join(cdir, stem + ".1.Inv")
                pea = os.path.join(cdir, stem + ".1.pea")
                if not all(os.path.exists(p) for p in (fid, inv, pea)):
                    continue
                seq, te, nech = "?", "?", 0
                for line in open(os.path.join(cdir, fn),
                                 encoding="gbk", errors="ignore"):
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == "SeqName":
                        seq = parts[1]
                    elif parts[0] == "TE(ms)":
                        te = parts[1]
                    elif parts[0] == "NECH":
                        nech = parts[1]
                records.append({
                    "campaign": campaign, "cond": cond_dir, "stem": stem,
                    "dir": cdir, "fid": fid, "inv": inv, "pea": pea,
                    "te": float(te), "seq": seq, "nech": int(nech),
                })

    if len(records) != 35:
        raise ValueError(f"Expected the same 35 records, got {len(records)}")
    if len({(r['campaign'], r['stem']) for r in records}) != 35:
        raise ValueError("Duplicate record identity")
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    historical = WORK_ROOT / "19_35条留出结果复核_20261025" / "holdout_review.json"
    old_manifest = json.loads(historical.read_text(encoding="utf-8"))["cohort_manifest"]
    original_by_name = {(r['campaign'], r['name']): r for r in old_manifest}
    for r in records:
        old = original_by_name[(r['campaign'], r['stem'])]
        for field in ("fid", "pea", "inv"):
            if sha(r[field]) != old[field + "_sha256"]:
                raise ValueError(f"Local data changed: {r['stem']} {field}")
    tracked = [Path(__file__), historical,
               Path(H.MODEL_DIR) / "h1_T06_退火长训.pth",
               Path(H.MODEL_DIR) / "h1_final_seed3.pth",
               Path(H.RESULT_DIR) / "classical_params_masked_20261025.json",
               Path(__file__).parent / "35条冻结留出验证结果.csv",
               Path(__file__).parent / "35条冻结留出_v2_统一指标.csv"]
    tracked += [Path(r[k]) for r in records for k in ("fid", "pea", "inv")]
    before = {str(p): sha(p) for p in tracked}
    manifest = {"written_before_this_reanalysis": datetime.now(timezone.utc).isoformat(),
                "role": "post-result protocol correction, not a new blind holdout",
                "protocols": {"pea": "actual .pea columns time_ms/amp; interpolate; zero missing tail; first-point normalization",
                              "raw_mag": "actual .Fid hypot(I,Q) on col4 times; same resampling/missing-tail/normalization"},
                "baseline_parameters": {"direct_lambda": lam,
                    "SG_old": [31, 2, 50.0], "SG_ext": [sg_win, sg_poly, sg_lam]},
                "records": records, "source_sha256": before}
    (args.output_dir / "protocol_and_inputs_before_run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    mA_old = load_m("h1_T06_退火长训.pth")
    mA_fin = load_m("h1_final_seed3.pth")
    print(f"同一批历史记录: {len(records)} 条；本地 Fid/pea/Inv 与既有指纹全匹配", flush=True)

    # 逐条: 两种输入协议 × 5 分支
    rows = []
    for r in records:
        t_raw, mag = load_fid_mag(r["fid"])
        # pea 输入协议（与既有 15 条一致）
        pea = _parse_columns(r["pea"], 4)
        if pea.ndim != 2 or not len(pea):
            raise ValueError(f"Empty pea: {r['pea']}")
        pea = pea[np.argsort(pea[:, 0])]
        if not np.all(np.diff(pea[:, 0]) > 0):
            raise ValueError(f"Non-increasing pea times: {r['pea']}")
        pea_env = np.interp(STD_ECHO_TIMES, pea[:, 0], pea[:, 1])
        v_pea = (STD_ECHO_TIMES <= pea[-1, 0]).astype(np.float32)
        pea_env[v_pea < 0.5] = 0.0
        pea_env = (pea_env / pea_env[0]).astype(np.float32)

        # 原始模值直接插值协议（frozen_holdout_35.py 原始口径）
        raw_env = np.interp(STD_ECHO_TIMES, t_raw, mag)
        v_raw = (STD_ECHO_TIMES <= t_raw[-1]).astype(np.float32)
        raw_env[v_raw < 0.5] = 0.0
        raw_env = (raw_env / raw_env[0]).astype(np.float32)
        for env in (pea_env, raw_env):
            if not np.all(np.isfinite(env)):
                raise ValueError(f"Nonfinite input: {r['stem']}")

        # 参考
        t2_ref, amp_ref = read_inv(r["inv"])
        ref = np.interp(STD_LOG_T2, np.log10(np.clip(t2_ref, 1e-6, None)),
                        amp_ref, left=0.0, right=0.0)
        ar = float(np.sum(ref) * (STD_LOG_T2[1] - STD_LOG_T2[0]))
        if ar <= 1e-9:
            continue
        ref = (ref / ar).astype(np.float32)

        for proto, env, vld in [("pea", pea_env, v_pea),
                                 ("raw_mag", raw_env, v_raw)]:
            sA_old = predict_spectra(mA_old, [env], torch.device("cpu"))[0]
            sA_fin = predict_spectra(mA_fin, [env], torch.device("cpu"))[0]
            nN = nnls_invert(env, lam, vld)
            nS_old = sg_nnls_invert(env, 31, 2, 50.0, vld)
            nS_ext = sg_nnls_invert(env, sg_win, sg_poly, sg_lam, vld)

            ref_pk = float(np.log10(t2_ref[np.argmax(amp_ref)]))
            for name, spec in [("A旧版", sA_old), ("A定稿", sA_fin),
                               ("直接NNLS", nN), ("SG旧", nS_old),
                               ("SG扩展", nS_ext)]:
                met = spectrum_metrics(spec, ref, env, vld)
                met["input_protocol"] = proto
                met["branch"] = name
                met["campaign"] = r["campaign"]
                met["样品"] = r["stem"]
                met["参考主峰ms"] = round(10 ** (ref_pk / 1), 1) if ref_pk > -5 else 0
                met["分支主峰ms"] = float(10 ** peak_pos_log(spec))
                met["参考网格主峰ms"] = float(10 ** peak_pos_log(ref))
                met["TE_ms"] = r["te"]
                met["NECH"] = r["nech"]
                met["有效点数"] = int(np.sum(vld))
                met["两输入相关性"] = pearson_r(pea_env, raw_env)
                rows.append(met)
        print(f"完成 {r['campaign']} / {r['stem']}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "35条冻结模型_协议修正_v3.csv",
              index=False, encoding="utf-8-sig")

    # 按协议×分支汇总
    for proto in ["pea", "raw_mag"]:
        sub = df[df["input_protocol"] == proto]
        cols = ["A旧版", "A定稿", "直接NNLS", "SG旧", "SG扩展"]
        print(f"\n==== {proto} 协议 ====")
        for br in cols:
            s = sub[sub["branch"] == br]
            print(f"  {br:12s}  r={s['pearson_r'].mean():.4f}  "
                  f"峰位={s['peak_err_log'].mean():.4f}  L2={s['spec_l2'].mean():.4f}")

    # 统计: A定稿 vs SG旧 的配对差
    print("\n==== A定稿 vs SG旧 配对差(全集) ====")
    for proto in ["pea", "raw_mag"]:
        sub = df[df["input_protocol"] == proto]
        a = sub[sub["branch"] == "A定稿"].set_index("样品")["pearson_r"]
        s = sub[sub["branch"] == "SG旧"].set_index("样品")["pearson_r"]
        common = a.index.intersection(s.index)
        d = (a[common] - s[common]).dropna()
        print(f"  {proto}: 平均Δr={d.mean():+.4f}  "
              f"SE={d.std()/np.sqrt(len(d)):.4f}  n={len(d)}")
    unchanged = before == {str(p): sha(p) for p in tracked}
    (args.output_dir / "completion.json").write_text(json.dumps(
        {"n_records": len(records), "n_metric_rows": len(rows),
         "sources_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("Tracked source changed during run")


if __name__ == "__main__":
    main()
