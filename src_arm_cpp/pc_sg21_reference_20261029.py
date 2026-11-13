# -*- coding: utf-8 -*-
"""
pc_sg21_reference_20261029.py —— 同配置 SG(21,2) 双通道实验：PC 参考输出生成

作用（不需要板子，上电前即可跑完）：
  对 15 条主数据，走与质量实验（cascade_lowsnr.py）完全相同的数据链——
  load_fid → align_to_50k → savgol_filter(ch, 21, 2) 逐通道（α=0 洁净档）——
  生成 PC 参考滤波输出，供板端输出回传后逐位 diff。

  另做一项输入锚定检查：把 data_up（板子实际吃的字节，01_准备上传数据.py 生成）
  与本脚本重走的 50kHz 对齐链逐点比对，确认"板子输入 = 质量实验输入"。
  若有差异（理论上应为 0 或 float32 量化级），在报告里醒目标出。

输出（本目录）：
  pc_ref/{name}.sg21pc.iq.bin —— float32 交错 I/Q 参考输出（scipy 口径）
  pc_ref_summary.json         —— 每条记录的 n_pts/te/split/sha256/输入锚定差
  pc_ref_report.md            —— 人读报告

运行环境：Python 3.12（与级联实验同一环境）
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_BUNDLE_ROOT, "src_host")
sys.path.insert(0, _SRC)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# —— Portable path resolution with optional environment overrides ——
V4_DIR = os.environ.get("LFNMR_V4_DIR", os.path.join(_BUNDLE_ROOT, "external_tools", "portable_nmr_eval_v4"))
sys.path.insert(0, V4_DIR)

from realdata import load_real_records  # noqa: E402
import hconfig as H  # noqa: E402
from legacy_io import load_fid  # noqa: E402
from timing_alignment import align_to_50k  # noqa: E402

DATA_UP = os.environ.get("LFNMR_BOARD_REPLAY_DATA_DIR", os.path.join(_BUNDLE_ROOT, "external_data", "board_replay_up"))
OUT = os.path.join(_HERE, "pc_ref")
os.makedirs(OUT, exist_ok=True)

WIN, ORDER = 21, 2  # 冻结配置：双通道 SG(21,2)，scipy 默认 mode='interp'


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    records = load_real_records()
    mdf = pd.read_csv(H.SPLIT_MANIFEST, encoding="utf-8-sig")
    mdf = mdf[mdf["seq_name"] == "CPMG"].set_index("sample_name")

    summary = []
    for r in records:
        name = r["name"]
        row = mdf.loc[name]
        stem = str(row["par_path"])[:-4]
        te_ms = float(row["te_ms"])

        # 1) 与质量实验同链：100kHz → 50kHz 对齐
        iq_100k, time_100k, _ = load_fid(stem + ".1.Fid")
        iq_50k, time_50k, _ = align_to_50k(iq_100k, time_100k)
        iq_50k = np.asarray(iq_50k)
        n = iq_50k.shape[1]

        # 2) PC 参考输出：scipy 逐通道 SG(21,2)（α=0 洁净档，mode='interp' 默认）
        sg = np.zeros_like(iq_50k)
        sg[0] = savgol_filter(iq_50k[0], WIN, ORDER)
        sg[1] = savgol_filter(iq_50k[1], WIN, ORDER)

        # 3) 输入锚定：data_up 字节（板上实际输入）vs 本链输出
        up_path = os.path.join(DATA_UP, f"{name}.iq.bin")
        up = np.fromfile(up_path, dtype=np.float32)
        fresh32 = np.empty(2 * n, dtype=np.float32)
        fresh32[0::2] = np.asarray(iq_50k[0], dtype=np.float32)
        fresh32[1::2] = np.asarray(iq_50k[1], dtype=np.float32)
        if up.size == fresh32.size:
            input_maxdiff = float(np.max(np.abs(up - fresh32)))
        else:
            input_maxdiff = float("nan")
            log(f"[WARN] {name}: data_up 大小 {up.size} != 重走链 {fresh32.size}")

        # 4) PC 参考输出落在 data_up 字节上（保证与板子同一输入）
        up2 = up.reshape(-1, 2).T if up.size == 2 * n else None
        if up2 is not None and input_maxdiff == 0.0:
            ref_in = up2  # 与重走链逐位一致，用谁都是同输入
        else:
            ref_in = np.asarray(iq_50k, dtype=np.float32)
            if input_maxdiff != 0.0:
                log(f"[WARN] {name}: 输入锚定差 {input_maxdiff:.3e}，"
                    f"PC 参考改用重走链输入（diff 时以本输出为准并注明）")
        ref = np.zeros_like(ref_in)
        ref[0] = savgol_filter(ref_in[0], WIN, ORDER)
        ref[1] = savgol_filter(ref_in[1], WIN, ORDER)

        inter = np.empty(2 * ref.shape[1], dtype=np.float32)
        inter[0::2] = ref[0]
        inter[1::2] = ref[1]
        out_path = os.path.join(OUT, f"{name}.sg21pc.iq.bin")
        inter.tofile(out_path)

        summary.append({
            "name": name, "split": r.get("split", ""), "te_ms": te_ms,
            "n_pts": int(ref.shape[1]),
            "input_anchor_maxdiff": input_maxdiff,
            "ref_sha256": sha256_file(out_path),
            "up_sha256": sha256_file(up_path),
        })
        log(f"{name:<12} n={ref.shape[1]} anchor_diff={input_maxdiff:.3e} 完成")

    with open(os.path.join(_HERE, "pc_ref_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    lines = [
        "# PC 参考输出生成报告（同配置 SG(21,2) 双通道，2026-10-29）", "",
        f"- 冻结配置：scipy savgol_filter(ch, {WIN}, {ORDER}) 逐通道，"
        "mode='interp'（默认）——与 cascade_lowsnr.py 质量实验 α=0 档同一代码路径；",
        f"- 15/15 条生成完毕，输出在 pc_ref/*.sg21pc.iq.bin；",
        "- 输入锚定（data_up 板上实际输入 vs 本脚本重走 50kHz 链）：",
    ]
    bad = [s for s in summary if s["input_anchor_maxdiff"] != 0.0]
    lines.append(f"  - 逐位一致 {len(summary)-len(bad)}/15；"
                 + (f"异常：{[s['name'] for s in bad]}" if bad else "无异常。"))
    lines += ["", "| 记录 | split | n_pts | 输入锚定 maxdiff | 参考输出 SHA256 前 12 位 |",
              "|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['name']} | {s['split']} | {s['n_pts']} | "
                     f"{s['input_anchor_maxdiff']:.3e} | {s['ref_sha256'][:12]} |")
    with open(os.path.join(_HERE, "pc_ref_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("报告写入 pc_ref_report.md；全部完成。")


if __name__ == "__main__":
    main()
