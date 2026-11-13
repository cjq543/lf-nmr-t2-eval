# -*- coding: utf-8 -*-
"""delta_main15_check.py —— 15条主数据的 .Inv/.pea 同款 δ 检验"""
import os, re, sys, glob
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_lime_experiment import delta_scan, DELTAS  # 复用同一机器

_BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NIUMAG_ROOT = os.environ.get("LFNMR_NIUMAG_DIR", os.path.join(_BUNDLE_ROOT, "external_data", "benchtop_niumag"))
OUT = os.path.join(_BUNDLE_ROOT, "results", "delta_scale_audit")

# 15 条主数据（16 总库 01/02 分库）的 pea+Inv 配对
pairs = {}
for pat in [os.path.join(_NIUMAG_ROOT, "01_180C_soybean_safflower", "**", "*.pea"),
            os.path.join(_NIUMAG_ROOT, "01_180C_soybean_safflower", "**", "*.Inv"),
            os.path.join(_NIUMAG_ROOT, "02_ambient24h_flax_safflower_rapeseed_sunflower", "**", "*.pea"),
            os.path.join(_NIUMAG_ROOT, "02_ambient24h_flax_safflower_rapeseed_sunflower", "**", "*.Inv")]:
    for p in glob.glob(pat, recursive=True):
        m = re.match(r"(.*?)(\.\d+)?\.(pea|Inv)$", p)
        if m and "冲突" not in p:
            pairs.setdefault((m.group(1), m.group(3)), p)

done = set()
rows = []
for (base, kind), p in sorted(pairs.items()):
    if kind != "Inv" or base in done:
        continue
    pk = (base, "pea")
    if pk not in pairs:
        continue
    t, amp = [], []
    for line in open(pairs[pk], encoding="gbk", errors="ignore"):
        parts = line.split()
        if len(parts) >= 2:
            try:
                t.append(float(parts[0])); amp.append(float(parts[1]))
            except ValueError:
                pass
    grid, amps = [], []
    for line in open(p, encoding="gbk", errors="ignore"):
        parts = line.split()
        if len(parts) >= 2:
            try:
                grid.append(float(parts[0])); amps.append(float(parts[1]))
            except ValueError:
                pass
    t, amp, grid, amps = np.array(t), np.array(amp), np.array(grid), np.array(amps)
    if len(t) < 50 or len(grid) < 10:
        continue
    d, r0, rs, ratio = delta_scan(t, amp, grid, amps)
    rec = os.path.basename(base)
    rows.append((rec, len(t), len(grid), d, r0, rs, ratio))
    done.add(base)
    print(f"  {rec}: pea{len(t)}点 Inv{len(grid)}点 δ*={d:+.4f} "
          f"resid {r0:.5f}→{rs:.5f} scale={ratio:.4f}")

with open(os.path.join(OUT, "results_main15_delta.csv"), "w", encoding="utf-8-sig") as f:
    f.write("record,n_pea,n_grid,delta_star,resid0,resid_star,scale_factor\n")
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")
ds = np.array([r[3] for r in rows])
print(f"\n=== 主数据 δ* 汇总 (n={len(ds)}) ===")
print(f"中位 {np.median(ds):+.4f}  范围 [{ds.min():+.4f}, {ds.max():+.4f}]  "
      f"|δ|≤0.01 比例: {np.mean(np.abs(ds)<=0.01)*100:.0f}%")
