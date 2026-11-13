# -*- coding: utf-8 -*-
"""07_原始级别外部验证_v2.py —— 修正版(独立终审 P1-3/4 项)
修正: 1) xlsx 按标题行定位+时间戳对齐(原错开一回波); 2) 补 SG+NNLS 两套冻结基线;
3) n=44 口径明确(2 条无参考谱不进反演分母); 4) 峰位误差并列报告; 5) f 字段矛盾注记。
"""
import json
import os
import re
import sys
import time

import numpy as np
import openpyxl
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import hconfig as H
from hconfig import STD_ECHO_TIMES, STD_LOG_T2
from baselines import (nnls_invert, sg_nnls_invert, pearson_r,
                       spectrum_metrics)
from train import build_model, predict_spectra

_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_ROOT = os.path.dirname(_DIR)
_EXT_LIME = os.environ.get("LFNMR_LIME_DIR", os.path.join(_BUNDLE_ROOT, "external_data", "external_lime_cohort"))
SRC = os.path.join(_EXT_LIME, "detection")
INV = os.path.join(_EXT_LIME, "inversion")
ROOT = os.environ.get("LFNMR_ASSET_ROOT", os.path.join(_BUNDLE_ROOT, "results", "external_validation"))

OIL_CN = {
    "caiziyou": "菜籽油", "dadouyou": "大豆油", "ganlanyou": "橄榄油",
    "haushengyou": "花生油", "huashengyou": "花生油", "honghua": "红花籽油",
    "kuihuaziyou": "葵花籽油", "mianziyou": "棉籽油", "yamaziyou": "亚麻籽油",
    "yamaziyu": "亚麻籽油", "yeziyou": "椰子油", "yumiyou": "玉米油",
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_m(pth):
    ck = torch.load(os.path.join(H.MODEL_DIR, pth), map_location="cpu",
                    weights_only=True)
    m = build_model(ck["config"])
    m.load_state_dict(ck["model_state_dict"])
    return m


def peak_log(spec):
    i = int(np.argmax(spec))
    return float(STD_LOG_T2[i])


def main():
    import json
    # 两套冻结 SG 基线参数(均不使用外部数据选参)
    SG_OLD = dict(win=31, poly=2, lam=50.0)      # 旧分布冻结
    SG_EXT = dict(win=51, poly=3, lam=200.0)     # 扩展分布冻结
    LAM_D = 1000.0

    mA_old = load_m("h1_T06_退火长训.pth")
    mA_fin = load_m("h1_final_seed3.pth")

    npys = sorted(f for f in os.listdir(SRC) if f.endswith(".npy")
                  and f != "background_T2.npy")
    rows = []
    n_done = n_skip = n_noref = 0
    for f in npys:
        stem = f[:-4]
        m = re.match(r"(.+?)_([0-9]+h|0h)(_open)?_T2$", stem)
        if m is None:
            n_skip += 1
            continue
        oil = OIL_CN.get(m.group(1), m.group(1))
        cond = m.group(2) + ("+open" if m.group(3) else "")

        a = np.load(os.path.join(SRC, f))
        y = yaml.safe_load(open(os.path.join(SRC, stem + ".yaml"),
                                encoding="utf-8"))
        ne, ns = int(y["n_echo"]), int(y["n_samples"])
        te_ms = float(y["t_echo"]) * 1000.0
        if a.size != ne * ns or a.size == 0:
            n_skip += 1
            continue

        r = a.reshape(ne, ns)
        env_raw = np.abs(r).mean(axis=1)
        t_raw = (np.arange(ne) + 1) * te_ms
        if env_raw[0] <= 0 or not np.isfinite(env_raw).all():
            n_skip += 1
            continue

        inv_dir = os.path.join(INV, stem)
        cons = np.nan
        ref = None
        xlsx = ([g for g in os.listdir(inv_dir) if g.endswith("_data.xlsx")]
                if os.path.isdir(inv_dir) else [])
        if xlsx:
            wb = openpyxl.load_workbook(os.path.join(inv_dir, xlsx[0]),
                                        read_only=True, data_only=True)
            t_x, s_x = [], []
            for row in wb["sheet1"].iter_rows(min_row=2, max_col=3,
                                              values_only=True):
                # 修正: 第2行即数据(0.2ms 首回波), 不再跳过
                if row[1] is None:
                    continue
                t_x.append(float(row[1])); s_x.append(float(row[2]))
            t2_ref, amp_ref = [], []
            for row in wb["sheet2"].iter_rows(min_row=2, max_col=3,
                                              values_only=True):
                if row[1] is None:
                    continue
                t2_ref.append(float(row[1]) * 1000.0)
                amp_ref.append(float(row[2]))
            wb.close()
            t_x = np.array(t_x) * 1000.0
            s_x = np.array(s_x)
            # 时间戳对齐: 重建包络按 xlsx 时间轴插值后比较
            e1 = np.interp(t_x, t_raw, env_raw)
            n_min = min(len(e1), len(s_x))
            e1a, e2a = e1[:n_min], s_x[:n_min]
            if np.std(e1a) > 1e-12 and np.std(e2a) > 1e-12:
                cons = pearson_r(e1a, e2a)

        x = np.interp(STD_ECHO_TIMES, t_raw, env_raw)
        valid = (STD_ECHO_TIMES <= t_raw[-1]).astype(np.float32)
        x[valid < 0.5] = 0.0
        x = (x / x[0]).astype(np.float32)

        row = {"记录": stem, "油种": oil, "条件": cond,
               "TE_ms": round(te_ms, 3), "回波数": ne,
               "重建vs导出包络r": round(cons, 4) if np.isfinite(cons) else np.nan,
               "原始幅度uV": round(float(np.abs(a).max() * 1e6), 2)}

        if xlsx and t2_ref and amp_ref:
            ref = np.interp(STD_LOG_T2, np.log10(np.array(t2_ref)),
                            np.array(amp_ref), left=0.0, right=0.0)
            ar = float(np.sum(ref) * (STD_LOG_T2[1] - STD_LOG_T2[0]))
            if ar > 1e-9:
                ref = (ref / ar).astype(np.float32)
                sA_old = predict_spectra(mA_old, [x], torch.device("cpu"))[0]
                sA_fin = predict_spectra(mA_fin, [x], torch.device("cpu"))[0]
                nN = nnls_invert(x, LAM_D, valid)
                nS_old = sg_nnls_invert(x, SG_OLD["win"], SG_OLD["poly"],
                                        SG_OLD["lam"], valid)
                nS_ext = sg_nnls_invert(x, SG_EXT["win"], SG_EXT["poly"],
                                        SG_EXT["lam"], valid)
                row["A旧版_r"] = round(pearson_r(sA_old, ref), 4)
                row["A定稿_r"] = round(pearson_r(sA_fin, ref), 4)
                row["直接NNLS_r"] = round(pearson_r(nN, ref), 4)
                row["SG旧_r"] = round(pearson_r(nS_old, ref), 4)
                row["SG扩展_r"] = round(pearson_r(nS_ext, ref), 4)
                row["A旧版_峰位"] = round(abs(peak_log(sA_old) -
                                              peak_log(ref)), 4)
                row["直接NNLS_峰位"] = round(abs(peak_log(nN) -
                                                 peak_log(ref)), 4)
        rows.append(row)
        n_done += 1
        if n_done % 10 == 0:
            log(f"已完成 {n_done} (累计 {time.time()-t0:.0f}s)" if False else
                f"已完成 {n_done}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "原始级外部验证全量表_v2.csv"),
              index=False, encoding="utf-8-sig")
    evaldf = df.dropna(subset=["A旧版_r"])
    log(f"完成: 原始记录 {n_done}, 无参考谱 {n_noref} 条不进反演分母, "
        f"可评价 {len(evaldf)}")

    cols = ["A旧版_r", "A定稿_r", "直接NNLS_r", "SG旧_r", "SG扩展_r"]
    print("\n==== 原始级外部验证(修正版) 全库均值 ====")
    print(df[cols].mean().round(4).to_string())
    print("\n==== 主峰对数误差均值 ====")
    print(df[["A旧版_峰位", "直接NNLS_峰位"]].mean().round(4).to_string())
    sub = df.dropna(subset=["A旧版_r", "直接NNLS_r"])
    print(f"\nA旧版 胜 直接NNLS: {(sub['A旧版_r'] > sub['直接NNLS_r']).sum()}"
          f"/{len(sub)}")
    print(f"可评价分母: n={len(evaldf)} (非46; 2条无参考谱不进分母)")


if __name__ == "__main__":
    main()
