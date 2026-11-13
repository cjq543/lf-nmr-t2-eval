# -*- coding: utf-8 -*-
"""
realdata.py —— 只读加载 15 份纽迈真实食用油 CPMG 记录(固定 9/3/3 划分)

数据来源: 外部纽迈原始采集归档 (经 LFNMR_RAW_DATA_ROOT 配置; 原始 .pea 回波包络 + .Inv 厂商反演谱)
划分清单: phase2_fixed_data_split_manifest.csv (Seed=42 整次采集划分, 沿用不动)

归一化约定(与 simulator.py 一致):
  * 包络: .pea 时间轴线性重采样到 STD_ECHO_TIMES(1024 点, 1~3000 ms), 除以首点;
  * 参考谱: .Inv 双列 (T2_ms, 幅度) 插值到 STD_LOG_T2(128 点), 归一到 log 面积=1;
  * 注意: .Inv 是纽迈厂商算法的参考输出, 本包一律称"厂商参考", 不称真值。
"""

import os
import numpy as np
import pandas as pd

from hconfig import (
    SPLIT_MANIFEST, STD_ECHO_TIMES, STD_LOG_T2, trapz_log,
)


def _parse_columns(path, n_cols, encoding="gbk"):
    recs = []
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < n_cols:
                continue
            try:
                recs.append([float(p) for p in parts[:n_cols]])
            except ValueError:
                continue
    return np.asarray(recs, dtype=np.float64)


def load_real_records():
    """返回 list[dict]: name/split/echo(1024)/spec(128)/scale。只读, 失败记录跳过。"""
    df = pd.read_csv(SPLIT_MANIFEST, encoding="utf-8-sig")
    df = df[df["seq_name"] == "CPMG"].copy()

    def split_of(tag):
        if tag.startswith("Train"):
            return "Train9"
        if tag.startswith("Val"):
            return "Val3"
        if tag.startswith("Test"):
            return "Test3"
        return "Other"

    records = []
    for _, row in df.iterrows():
        # 双轨路径自适应查找: 优先本地自包含文件 -> 其次外部原路径 -> 再次按 RAW_DATA_ROOT 拼接
        par_p = None
        for candidate in [row.get("local_par_path"), row.get("par_path")]:
            if candidate and os.path.exists(candidate):
                par_p = candidate
                break
        if par_p is None:
            sample = row["sample_name"]
            par_p = os.path.join(RAW_DATA_ROOT, row["rel_dir"], sample + ".par")

        base = par_p[:-4]
        pea_path = base + ".1.pea"
        inv_path = base + ".1.Inv"
        if not (os.path.exists(pea_path) and os.path.exists(inv_path)):
            # 若 local 对应缺少 pea/inv，回退外部原 base 尝试
            orig_base = str(row.get("par_path", ""))[:-4]
            if os.path.exists(orig_base + ".1.pea") and os.path.exists(orig_base + ".1.Inv"):
                base = orig_base
                pea_path = base + ".1.pea"
                inv_path = base + ".1.Inv"
            else:
                continue

        pea = _parse_columns(pea_path, 4)   # time_ms, amp, real, imag
        inv = _parse_columns(inv_path, 2)   # t2_ms, amp
        if pea.size == 0 or inv.size == 0:
            continue

        # 包络 -> 1024 点标准网格 + 观测有效窗 mask
        # (v3协议: 1800回波记录在 1800~3000ms 区间属缺失观测而非测得的零,
        #  mask=0 区域不参与物理前向损失/前向指标; 输入仍补零以维持网格形状)
        t, amp = pea[:, 0], pea[:, 1]
        order = np.argsort(t)
        t, amp = t[order], amp[order]
        echo = np.interp(STD_ECHO_TIMES, t, amp)
        valid = (STD_ECHO_TIMES <= t[-1]).astype(np.float32)
        echo[valid < 0.5] = 0.0
        scale = float(echo[0])
        if scale <= 1e-9:
            scale = float(np.max(echo)) if np.max(echo) > 0 else 1.0
        echo_n = echo / scale

        # 厂商谱 -> 128 点对数网格, 归一到单位 log 面积
        t2, spec = inv[:, 0], inv[:, 1]
        order = np.argsort(t2)
        t2, spec = t2[order], np.clip(spec[order], 0.0, None)
        log_spec = np.interp(STD_LOG_T2, np.log10(t2), spec, left=0.0, right=0.0)
        area = trapz_log(log_spec)
        spec_n = log_spec / area if area > 1e-9 else log_spec

        records.append({
            "name": str(row["sample_name"]),
            "split": split_of(str(row["dataset_split"])),
            "oil_type": str(row["oil_type"]),
            "heat": str(row["heat_condition"]),
            "te_ms": float(row["te_ms"]),
            "nech": int(row["nech"]),
            "echo": echo_n.astype(np.float32),
            "valid_mask": valid,
            "spec": spec_n.astype(np.float32),
            "raw_first_echo": scale,
        })
    return records


if __name__ == "__main__":
    recs = load_real_records()
    from collections import Counter
    print(len(recs), Counter(r["split"] for r in recs))
    for r in recs[:3]:
        print(r["name"], r["split"], r["echo"].shape, r["spec"].shape,
              round(float(r["echo"][0]), 4), round(trapz_log(r["spec"]), 4))
