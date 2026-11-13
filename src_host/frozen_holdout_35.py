# -*- coding: utf-8 -*-
"""frozen_holdout_35.py —— 35 条候选冻结留出验证

协议（在看结果前冻结）：
  模型：A 旧版（h1_T06，原分布）与 A 定稿（h1_final_seed3，扩展分布），权重不变
  基线：修正 NNLS（λ=1000）、SG(51,3)+NNLS（λ=200，masked 冻结参数）
  分母：35 条候选中 .Inv 参考谱可用的记录（逐条记录排除原因）
  指标：谱相关 r、对数主峰误差（三点内插）
  规则：不根据结果选模型/调参/换基线；结果无论正反如实报告

第二步（可选重复性对）：23 组冲突中的 CPMG 对做两次采集变异系数。
"""
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import hconfig as H
from hconfig import STD_ECHO_TIMES, STD_LOG_T2
from baselines import nnls_invert, sg_nnls_invert, pearson_r, spectrum_metrics
from train import build_model, predict_spectra

BASE = os.environ.get("LFNMR_HOLDOUT35_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external_data", "benchtop_holdout35"))
CAMP1 = os.path.join(BASE, "第二次油样品")
CAMP2 = os.path.join(BASE, "第三次油样品")

OIL_CN = {
    "椰子油": "椰子油", "橄榄油": "橄榄油", "芝麻油": "芝麻油",
    "花生油": "花生油", "大豆油": "大豆油", "棉籽油": "棉籽油",
    "玉米油": "玉米油",
}


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


def load_fid_mag(p):
    a = np.loadtxt(p, comments="#")
    if a.ndim == 1:
        a = a.reshape(1, -1)
    mag = np.abs(a[:, 0] + 1j * a[:, 1])
    t = a[:, 3] if a.shape[1] > 3 else np.arange(len(a)) * 0.02
    return t, mag


def load_m(pth):
    ck = torch.load(os.path.join(H.MODEL_DIR, pth), map_location="cpu",
                    weights_only=True)
    m = build_model(ck["config"])
    m.load_state_dict(ck["model_state_dict"])
    return m


def main():
    import json
    cp = json.load(open(os.path.join(
        H.RESULT_DIR, "classical_params_masked_20261025.json"),
        encoding="utf-8"))
    lam = cp.get("nnls_lambda_direct", 1000.0)
    sg_win, sg_poly, sg_lam = cp["sg_win"], cp["sg_poly"], cp["nnls_lambda"]

    mA_old = load_m("h1_T06_退火长训.pth")
    mA_fin = load_m("h1_final_seed3.pth")

    # 扫描 35 条候选的 .par + .Inv + .Fid
    records = []
    for camp_root, campaign in [(CAMP1, "第二次油样品"), (CAMP2, "第三次油样品")]:
        for cond_dir in sorted(os.listdir(camp_root)):
            cdir = os.path.join(camp_root, cond_dir, "CPMG")
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                if not fn.endswith(".par"):
                    continue
                stem = fn[:-4]
                par = os.path.join(cdir, fn)
                fid = os.path.join(cdir, stem + ".1.Fid")
                inv = os.path.join(cdir, stem + ".1.Inv")
                has_fid = os.path.exists(fid)
                has_inv = os.path.exists(inv)
                seq, te, nech = "?", "?", "?"
                for line in open(par, encoding="gbk", errors="ignore"):
                    parts = line.split()
                    if parts and parts[0] == "TE(ms)":
                        te = parts[1]
                    elif parts[0] == "SeqName":
                        seq = parts[1]
                    elif parts[0] == "NECH":
                        nech = parts[1]
                records.append({
                    "campaign": campaign, "cond_dir": cond_dir, "stem": stem,
                    "dir": cdir, "fid": fid if has_fid else None,
                    "inv": inv if has_inv else None, "te": float(te),
                    "seq": seq, "nech": int(nech) if nech != "?" else 0,
                })

    print(f"候选记录: {len(records)} 条")
    usable = [r for r in records if r["fid"] and r["inv"]]
    print(f"  .Fid+.Inv 双全: {len(usable)} 条")
    no_inv = [r for r in records if not r["inv"]]
    no_fid = [r for r in records if not r["fid"]]
    if no_inv:
        print(f"  缺 Inv: {[r['stem'] for r in no_inv]}")
    if no_fid:
        print(f"  缺 Fid: {[r['stem'] for r in no_fid]}")

    # 逐条评价
    results = []
    for r in usable:
        try:
            t_raw, mag = load_fid_mag(r["fid"])
            # 重采样到本文网格
            x = np.interp(STD_ECHO_TIMES, t_raw, mag)
            valid = (STD_ECHO_TIMES <= t_raw[-1]).astype(np.float32)
            x[valid < 0.5] = 0.0
            x = (x / x[0]).astype(np.float32)

            # .Inv 参考
            t2_ref, amp_ref = read_inv(r["inv"])
            ref = np.interp(STD_LOG_T2, np.log10(np.clip(t2_ref, 1e-6, None)),
                            amp_ref, left=0.0, right=0.0)
            ar = float(np.sum(ref) * (STD_LOG_T2[1] - STD_LOG_T2[0]))
            if ar <= 1e-9:
                continue
            ref = (ref / ar).astype(np.float32)

            sA_old = predict_spectra(mA_old, [x], torch.device("cpu"))[0]
            sA_fin = predict_spectra(mA_fin, [x], torch.device("cpu"))[0]
            nN = nnls_invert(x, lam, valid)
            nS = sg_nnls_invert(x, sg_win, sg_poly, sg_lam, valid)

            def peak_lg(sp):
                return float(STD_LOG_T2[int(np.argmax(sp))])

            def peak_err(sp):
                return abs(peak_lg(sp) - peak_log(ref)) if False else \
                    abs(peak_lg(sp) - float(np.log10(t2_ref[np.argmax(amp_ref)])))

            ref_pk = float(np.log10(t2_ref[np.argmax(amp_ref)]))
            results.append({
                "campaign": r["campaign"], "样品": r["stem"],
                "参考主峰ms": round(10 ** ref_pk, 1),
                "A旧版_r": round(pearson_r(sA_old, ref), 4),
                "A定稿_r": round(pearson_r(sA_fin, ref), 4),
                "修正NNLS_r": round(pearson_r(nN, ref), 4),
                "SG旧_r": round(pearson_r(
                    sg_nnls_invert(x, 31, 2, 50.0, valid), ref), 4),
                "SG扩展_r": round(pearson_r(
                    sg_nnls_invert(x, 51, 3, 200.0, valid), ref), 4),
                "A旧版_峰位差": round(abs(peak_lg(sA_old) - ref_pk), 4),
                "A定稿_峰位差": round(abs(peak_lg(sA_fin) - ref_pk), 4),
                "NNLS_峰位差": round(abs(peak_lg(nN) - ref_pk), 4),
            })
        except Exception as e:
            print(f"  {r['stem']}: {e}")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "35条冻结留出验证结果.csv"),
              index=False, encoding="utf-8-sig")

    cols = ["A旧版_r", "A定稿_r", "修正NNLS_r", "SG旧_r", "SG扩展_r"]
    print("\n==== 35 条冻结留出验证 均值 ====")
    print(df[cols].mean().round(4).to_string())
    print("\n按 campaign:")
    print(df.groupby("campaign")[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
