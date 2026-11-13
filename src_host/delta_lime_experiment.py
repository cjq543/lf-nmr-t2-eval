# -*- coding: utf-8 -*-
"""delta_lime_experiment.py —— 尺度异常 δ 专项：LIME 46 条同款检验
步骤：
  A. 机制验证：对 21 号已知 δ 的纽迈记录（椰子油48h/橄榄油0h）用本实现复算，
     应得 δ≈−0.17（验证实验机器正确）；
  B. LIME 检验：46 条外部记录（厂商包络 sheet1 + 厂商参考谱 sheet2，原生网格，
     LS 幅度对齐）逐条扫 δ，看最优 δ 分布。
判读：LIME δ≈0 → 异常 localize 在纽迈参考流水线；δ≈−0.17 → 共享约定问题。
输出：本目录 results_lime_delta.csv + results_niumag_validation.csv
"""
import os, re, sys, glob
import numpy as np
import openpyxl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXT_ROOT = os.environ.get("LFNMR_RAW_DATA_ROOT", os.path.join(_BUNDLE_ROOT, "external_data"))
OUT = os.path.join(_BUNDLE_ROOT, "results", "delta_scale_audit")
os.makedirs(OUT, exist_ok=True)
DELTAS = np.arange(-0.60, 0.6001, 0.0025)


def delta_scan(t_ms, sig, grid_ms, amps, sub=1):
    """原生网格前向 + LS幅度对齐 + δ 扫描。返回 (delta*, resid0, resid*, ratio*)"""
    if sub > 1:
        t_ms, sig = t_ms[::sub], sig[::sub]
    s = np.asarray(sig, float)
    s = s / np.linalg.norm(s)
    E = np.exp(-np.outer(t_ms, 1.0 / np.asarray(grid_ms, float)))  # n_t × n_grid
    best = (None, 1e9, None)
    res0 = None
    for d in DELTAS:
        c = 10.0 ** (-d)          # T2' = T2*10^d  <=>  exp(-t/T2') = E^c
        y = (E ** c) @ np.asarray(amps, float)
        ny = np.linalg.norm(y)
        if ny <= 0:
            continue
        a = float(np.dot(y, s)) / (ny * ny)
        r = np.linalg.norm(s - a * y)
        if abs(d) < 1e-12:
            res0 = r
        if r < best[1]:
            best = (float(d), float(r), float(10 ** d))
    return best[0], res0, best[1], best[2]


# ---------- A. 纽迈机制验证 ----------
print("=== A. 纽迈机制验证（21号已知 δ）===")
NIU = []
for pat in [os.path.join(_EXT_ROOT, "benchtop_niumag", "**", "椰子油48h*"),
            os.path.join(_EXT_ROOT, "benchtop_niumag", "**", "橄榄油0h*")]:
    for p in glob.glob(pat, recursive=True):
        if p.endswith((".pea", ".Inv")) and "冲突" not in p:
            NIU.append(p)
print("找到文件:", len(NIU))
val_rows = []
# 按 base 配对 .pea + .Inv
pairs = {}
for p in NIU:
    m = re.match(r"(.*?)(\.\d+)?\.(pea|Inv)$", p)
    if not m:
        continue
    base, inv = m.group(1), m.group(3)
    pairs.setdefault((base, inv), p)
for (base, kind), p in sorted(pairs.items()):
    rec = os.path.basename(base)
    if kind == "pea":
        t, amp = [], []
        for line in open(p, encoding="gbk", errors="ignore"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    t.append(float(parts[0])); amp.append(float(parts[1]))
                except ValueError:
                    continue
        pairs[(base, "pea_data")] = (np.array(t), np.array(amp))
for (base, kind), p in sorted(pairs.items()):
    if kind != "Inv":
        continue
    rec = os.path.basename(base)
    grid, amps = [], []
    for line in open(p, encoding="gbk", errors="ignore"):
        parts = line.split()
        if len(parts) >= 2:
            try:
                grid.append(float(parts[0])); amps.append(float(parts[1]))
            except ValueError:
                continue
    key = (base, "pea_data")
    if key not in pairs:
        print(f"  ! {rec}: 无配对 pea，跳过")
        continue
    t, amp = pairs[key]
    grid, amps = np.array(grid), np.array(amps)
    d, r0, rs, ratio = delta_scan(t, amp, grid, amps)
    val_rows.append((rec, len(t), len(grid), grid.max(), d, r0, rs, ratio))
    print(f"  {rec}: pea {len(t)}点(至{t.max():.1f}ms) Inv {len(grid)}点(至{grid.max():.1f}ms) "
          f"δ*={d:+.4f} residual {r0:.5f}→{rs:.5f} scale={ratio:.4f}")

with open(os.path.join(OUT, "results_niumag_validation.csv"), "w",
          encoding="utf-8-sig") as f:
    f.write("record,n_pea,n_grid,grid_max_ms,delta_star,resid0,resid_star,scale_factor\n")
    for r in val_rows:
        f.write(",".join(str(x) for x in r) + "\n")

# ---------- B. LIME 46 条 ----------
print("\n=== B. LIME 46 条 δ 检验 ===")
SRC = os.environ.get("LFNMR_LIME_INV_DIR", os.path.join(_EXT_ROOT, "external_lime_cohort", "inversion"))
rows = []
for dname in sorted(os.listdir(SRC)):
    full = os.path.join(SRC, dname)
    if not os.path.isdir(full) or not dname.endswith("_T2") or "_D" in dname:
        continue
    xl = [f for f in os.listdir(full) if f.endswith("_data.xlsx")]
    if not xl:
        continue
    wb = openpyxl.load_workbook(os.path.join(full, xl[0]), read_only=True,
                                data_only=True)
    t, sig, t2, amp = [], [], [], []
    for row in wb["sheet1"].iter_rows(min_row=3, max_col=3, values_only=True):
        if row[1] is None or row[2] is None:
            continue
        t.append(float(row[1]) * 1000.0); sig.append(float(row[2]))
    for row in wb["sheet2"].iter_rows(min_row=3, max_col=3, values_only=True):
        if row[1] is None or row[2] is None:
            continue
        t2.append(float(row[1]) * 1000.0); amp.append(float(row[2]))
    wb.close()
    t, sig, t2, amp = np.array(t), np.array(sig), np.array(t2), np.array(amp)
    if len(t) < 50 or len(t2) < 10:
        print(f"  ! {dname}: 数据不足 t={len(t)} grid={len(t2)}")
        continue
    d, r0, rs, ratio = delta_scan(t, sig, t2, amp)
    ref_peak = float(t2[int(np.argmax(amp))])
    rows.append((dname, len(t), t.min(), t.max(), len(t2), t2.min(), t2.max(),
                 ref_peak, d, r0, rs, ratio))
    print(f"  {dname}: 包络{len(t)}点({t.min():.2f}-{t.max():.1f}ms) 参考{len(t2)}点"
          f"(峰{ref_peak:.1f}ms) δ*={d:+.4f} resid {r0:.5f}→{rs:.5f} scale={ratio:.4f}")

with open(os.path.join(OUT, "results_lime_delta.csv"), "w", encoding="utf-8-sig") as f:
    f.write("record,n_env,t_min_ms,t_max_ms,n_grid,grid_min,grid_max,ref_peak_ms,"
            "delta_star,resid0,resid_star,scale_factor\n")
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")

# 汇总
ds = np.array([r[8] for r in rows])
print(f"\n=== LIME δ* 汇总 (n={len(ds)}) ===")
print(f"中位 {np.median(ds):+.4f}  IQR [{np.percentile(ds,25):+.4f}, "
      f"{np.percentile(ds,75):+.4f}]  范围 [{ds.min():+.4f}, {ds.max():+.4f}]")
print(f"|δ|≤0.01 的比例: {np.mean(np.abs(ds) <= 0.01)*100:.0f}%")
print(f"δ ∈ [-0.18,-0.16] 的比例: {np.mean((ds>=-0.18)&(ds<=-0.16))*100:.0f}%")
