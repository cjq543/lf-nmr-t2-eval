# -*- coding: utf-8 -*-
"""delta_offset_experiment.py —— 判别纽迈参考谱尺度异常的来源
模型：s(t) ≈ a·Σ_k m_k·exp(-t/(T2_k·10^δ)) + c   （a、c 对每个 δ 线性最小二乘）
判别：
  若加自由截距 c 后 δ* → 0 且 c ≈ .pea 尾部平台 ⇒ 厂商核含常数基线（约定差）
  若 δ* 不动 ⇒ 时间轴尺度约定差，继续查 t0 平移变体
对照：LIME 2 条同款检验（应保持 δ≈0）
"""
import os, re, sys, glob
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXT_ROOT = os.environ.get("LFNMR_RAW_DATA_ROOT", os.path.join(_BUNDLE_ROOT, "external_data"))
OUT = os.path.join(_BUNDLE_ROOT, "results", "delta_scale_audit")
DELTAS = np.arange(-0.60, 0.6001, 0.0025)


def load_niumag(rec_glob):
    pea = inv = None
    for p in glob.glob(rec_glob, recursive=True):
        m = re.match(r"(.*?)(\.\d+)?\.(pea|Inv)$", p)
        if not m or "冲突" in p:
            continue
        if m.group(3) == "pea":
            pea = p
        else:
            inv = p
    t, amp = [], []
    for line in open(pea, encoding="gbk", errors="ignore"):
        parts = line.split()
        if len(parts) >= 2:
            try:
                t.append(float(parts[0])); amp.append(float(parts[1]))
            except ValueError:
                pass
    grid, amps = [], []
    hdr = []
    for line in open(inv, encoding="gbk", errors="ignore"):
        parts = line.split()
        if len(parts) >= 2:
            try:
                grid.append(float(parts[0])); amps.append(float(parts[1]))
            except ValueError:
                hdr.append(line.strip()[:80])
        elif line.strip():
            hdr.append(line.strip()[:80])
    return np.array(t), np.array(amp), np.array(grid), np.array(amps), hdr[:6]


def scan(t, s, grid, amps, with_offset):
    """返回 (δ*, resid*, (a,c))；a,c 线性LS；resid=‖s-模型‖/‖s‖"""
    E = np.exp(-np.outer(t, 1.0 / np.asarray(grid, float)))
    s = np.asarray(s, float)
    nrm = np.linalg.norm(s)
    best = (None, 1e9, None)
    for d in DELTAS:
        c_ = 10.0 ** (-d)
        y = (E ** c_) @ np.asarray(amps, float)
        if with_offset:
            M = np.stack([y, np.ones_like(y)], axis=1)
            coef, *_ = np.linalg.lstsq(M, s, rcond=None)
            r = np.linalg.norm(M @ coef - s) / nrm
            ac = coef
        else:
            a = float(np.dot(y, s)) / np.dot(y, y)
            r = np.linalg.norm(a * y - s) / nrm
            ac = (a, 0.0)
        if r < best[1]:
            best = (float(d), float(r), ac)
    return best


def report(name, t, amp, grid, amps, hdr):
    a0 = amp[amp > 0][0]
    tail = amp[-max(1, len(amp) // 10):]
    print(f"\n===== {name} =====")
    print(f"  pea {len(t)}点 ({t.min():.2f}-{t.max():.1f}ms)  Inv {len(grid)}点 "
          f"({grid.min():.3g}-{grid.max():.4g})")
    if hdr:
        print(f"  Inv头部: {hdr}")
    print(f"  首幅={a0:.1f}  尾部10%均值={tail.mean():.1f} "
          f"(平台比 {tail.mean()/a0:.3f})  尾部10%std={tail.std():.1f}")
    d0, r0, _ = scan(t, amp, grid, amps, with_offset=False)
    d1, r1, (a, c) = scan(t, amp, grid, amps, with_offset=True)
    print(f"  无截距: δ*={d0:+.4f} resid={r0:.5f}")
    print(f"  带截距: δ*={d1:+.4f} resid={r1:.5f}  a={a:.4g} c={c:.4g} "
          f"(c/首幅={c/a0:.3f}, c/尾部平台={c/max(tail.mean(),1e-9):.2f})")
    return d0, r0, d1, r1, c / a0


# ===== 纽迈：2 留出 + 2 主数据 =====
res = {}
cases = [
    ("椰子油48h(留出)", os.path.join(_EXT_ROOT, "benchtop_niumag", "**", "椰子油48h*")),
    ("橄榄油0h(留出)", os.path.join(_EXT_ROOT, "benchtop_niumag", "**", "橄榄油0h*")),
    ("大豆油24h(主)", os.path.join(_EXT_ROOT, "benchtop_niumag", "01_180C_soybean_safflower", "**", "大豆油24h*")),
    ("菜籽油24h(主常温)", os.path.join(_EXT_ROOT, "benchtop_niumag", "02_ambient24h_flax_safflower_rapeseed_sunflower", "**", "菜籽油24h*")),
]
for name, pat in cases:
    try:
        t, amp, grid, amps, hdr = load_niumag(pat)
        if len(t) < 50:
            print(f"{name}: 数据不足")
            continue
        res[name] = report(name, t, amp, grid, amps, hdr)
    except Exception as e:
        print(f"{name}: 失败 {e}")

# ===== LIME 对照：2 条 =====
import openpyxl
SRC = os.environ.get("LFNMR_LIME_INV_DIR", os.path.join(_EXT_ROOT, "external_lime_cohort", "inversion"))
for dname in ["caiziyou_0h_T2", "ganlanyou_0h_T2"]:
    full = os.path.join(SRC, dname)
    xl = [f for f in os.listdir(full) if f.endswith("_data.xlsx")][0]
    wb = openpyxl.load_workbook(os.path.join(full, xl), read_only=True, data_only=True)
    t, sig, t2, amp = [], [], [], []
    for row in wb["sheet1"].iter_rows(min_row=3, max_col=3, values_only=True):
        if row[1] is not None and row[2] is not None:
            t.append(float(row[1]) * 1000.0); sig.append(float(row[2]))
    for row in wb["sheet2"].iter_rows(min_row=3, max_col=3, values_only=True):
        if row[1] is not None and row[2] is not None:
            t2.append(float(row[1]) * 1000.0); amp.append(float(row[2]))
    wb.close()
    res[dname] = report(dname + "(LIME)", np.array(t), np.array(sig),
                        np.array(t2), np.array(amp), [])
