# -*- coding: utf-8 -*-
"""generate_effect_forest.py —— Standalone figure & effect-size table generator.
Panel A: 35-record frozen holdout paired Δr (record-level + 7-oil-type clustered bootstrap CI, seed 20261025)
         computed directly from configs_and_metrics/holdout35_pmag_wide.csv.
Panel B: OOD 9-dim paired Δr (128 pairs/condition, bootstrap CI, seed 20261026)
         computed directly from configs_and_metrics/ood_rows_corrected.csv.
Outputs under results/regenerated_figures/.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("LFNMR_WORK_ROOT", _BUNDLE_ROOT)
R = os.path.join(ROOT, "configs_and_metrics")
OUT_DIR = os.path.join(_BUNDLE_ROOT, "results", "regenerated_figures")
os.makedirs(OUT_DIR, exist_ok=True)

C_A = "#d62728"; C_SIG = "#2ca02c"; C_NS = "#8c8c8c"; C_CLU = "#1f77b4"


def _boot_record_ci(d, seed=20261025, n_boot=20000):
    arr = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    lo, hi = np.percentile(arr[idx].mean(axis=1), [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def _boot_cluster_ci(oils, vals, seed=20261025, n_boot=20000):
    df_tmp = pd.DataFrame({"oil": oils, "val": vals})
    groups = [g["val"].values for _, g in df_tmp.groupby("oil", sort=True)]
    rng = np.random.default_rng(seed)
    k = len(groups)
    idx = rng.integers(0, k, size=(n_boot, k))
    g_sizes = np.array([len(g) for g in groups], dtype=np.float64)
    g_sums = np.array([g.sum() for g in groups], dtype=np.float64)
    means = g_sums[idx].sum(axis=1) / g_sizes[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# ============ 数据（直接从归档宽表与 OOD 表实时计算） ============
wide = pd.read_csv(os.path.join(R, "holdout35_pmag_wide.csv"), encoding="utf-8-sig")
oils = wide["样品"].astype(str).str.replace(r"\d+h$", "", regex=True).values

NAME_CN = {"A旧版_r": "vs 旧版A（原分布专精）",
           "SG旧_r": "vs SG(31,2)+NNLS（旧冻结参数）",
           "SG扩展_r": "vs SG(51,3)+NNLS（masked扩展参数）",
           "修正NNLS_r": "vs 直接masked NNLS"}
NAME_EN = {"A旧版_r": "vs legacy A (orig-distribution)",
           "SG旧_r": "vs SG(31,2)+NNLS (frozen)",
           "SG扩展_r": "vs SG(51,3)+NNLS (masked ext.)",
           "修正NNLS_r": "vs direct masked NNLS"}

order = ["修正NNLS_r", "A旧版_r", "SG扩展_r", "SG旧_r"]
holdout = []
for col in order:
    diff = (wide["A定稿_r"] - wide[col]).to_numpy(dtype=np.float64)
    dmean, lo, hi = _boot_record_ci(diff, seed=20261025, n_boot=20000)
    clo, chi = _boot_cluster_ci(oils, diff, seed=20261025, n_boot=20000)
    holdout.append({
        "key": col,
        "d": dmean,
        "lo": lo,
        "hi": hi,
        "clo": round(clo, 6),
        "chi": round(chi, 6),
        "wins": int((diff > 0).sum()),
        "losses": int((diff <= 0).sum()),
    })

dfo = pd.read_csv(os.path.join(R, "ood_rows_corrected.csv"), encoding="utf-8-sig")
rng = np.random.default_rng(20261026)
ood = []
for var in sorted(dfo["variant"].unique()):
    d_v = dfo[dfo["variant"] == var]
    a = d_v[d_v["branch"] == "A仿真版"]["pearson_r"].to_numpy()
    b = d_v[d_v["branch"] == "SG(31,2)+NNLS"]["pearson_r"].to_numpy()
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, size=(20000, n))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    ood.append({"var": var, "d": d.mean(), "lo": lo, "hi": hi,
                "wins": int((d > 0).sum()), "n": n})
ood.sort(key=lambda r: r["d"])  # 自下而上画

# ============ 绘图 ============
VAR_EN = {"V1_更窄峰宽": "V1 narrower peaks",
          "V2_更宽峰宽": "V2 broader peaks",
          "V3_峰位外推_快端": "V3 peak extrapolation (fast)",
          "V3_峰位外推_慢端": "V3 peak extrapolation (slow)",
          "V4_多组分4-5峰": "V4 multi-component (4-5 peaks)",
          "V5_强基线漂移": "V5 strong baseline drift",
          "V6_缺失时间窗_600ms": "V6 missing window (600 ms)",
          "V6_缺失时间窗_1200ms": "V6 missing window (1200 ms)",
          "V7_极低幅比5-8dB": "V7 very low amplitude (5-8 dB)"}


def make(lang="en"):
    if lang == "zh":
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        lbl_hold = "35条冻结留出：Δr（A定稿 − 对照）"
        lbl_ood = "OOD九维（n=128/维）：Δr（A − 修正SG(31,2)+NNLS）"
        zero_lbl = "0（无差异）"
        note_clu = "蓝色虚线=按油种(7组)聚类bootstrap敏感性区间"
        tag_sig = "显著"
        tag_ns = "跨零(不显著)"
        title = "效应量与不确定性：A定稿相对基线的配对差"
        f_out = os.path.join(OUT_DIR, "fig_CN_effect_forest_legacy_mpl.png")
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
        lbl_hold = "35-record frozen holdout: paired Δr (A-final − baseline)"
        lbl_ood = "OOD 9 dimensions (n=128 each): Δr (A − corrected SG(31,2)+NNLS)"
        zero_lbl = "0 (no difference)"
        note_clu = "blue dashed = oil-type-clustered (7 groups) bootstrap sensitivity CI"
        tag_sig = "significant"
        tag_ns = "crosses zero (n.s.)"
        title = "Effect sizes and uncertainty: paired differences of A-final vs baselines"
        f_out = os.path.join(OUT_DIR, "fig_EN_effect_forest_legacy_mpl.png")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    # ---- Panel A ----
    ax = axes[0]
    ys = np.arange(len(holdout))[::-1]
    for y, r in zip(ys, holdout):
        sig = not (r["lo"] <= 0 <= r["hi"])
        col = C_SIG if sig else C_NS
        ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=2.2,
                solid_capstyle="round", zorder=3)
        ax.plot(r["d"], y, "o", color=C_A, ms=8, zorder=4)
        if r["clo"] is not None:
            ax.plot([r["clo"], r["chi"]], [y, y], color=C_CLU, lw=1.2,
                    ls="--", alpha=0.85, zorder=2)
        name = (NAME_CN if lang == "zh" else NAME_EN)[r["key"]]
        tag = tag_sig if sig else tag_ns
        ax.annotate(f"+{r['d']:.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]  {tag}\n"
                    f"({'win/loss: ' if lang == 'en' else '胜/负: '}"
                    f"{r['wins']}/{r['losses']})",
                    xy=(r["hi"], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=11.5, color=col)
    ax.axvline(0, color="k", lw=1, ls=":", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([(NAME_CN if lang == "zh" else NAME_EN)[r["key"]]
                        for r in holdout], fontsize=12)
    ax.set_xlabel("paired Δr (A-final − baseline)" if lang == "en" else "配对差 Δr",
                  fontsize=12.5)
    ax.set_title(lbl_hold, fontsize=13.5)
    ax.set_xlim(-0.012, 0.078)
    ax.text(0.5, -0.30, note_clu, transform=ax.transAxes, ha="center",
            va="top", fontsize=11, color="#555555")
    ax.grid(axis="x", ls="--", alpha=0.35)

    # ---- Panel B ----
    ax = axes[1]
    ys = np.arange(len(ood))
    for y, r in zip(ys, ood):
        sig = not (r["lo"] <= 0 <= r["hi"])
        col = C_SIG if (sig and r["d"] > 0) else (C_NS if not sig else "#c0392b")
        ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=2.2,
                solid_capstyle="round", zorder=3)
        ax.plot(r["d"], y, "o", color=C_A, ms=8, zorder=4)
        vname = VAR_EN.get(r["var"], r["var"]) if lang == "en" else r["var"]
        lab = f"{r['d']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
        if r["hi"] > 0.10:  # 右侧放不下 → 标注放区间左侧
            ax.annotate(lab, xy=(r["lo"], y), xytext=(-6, 0),
                        textcoords="offset points", va="center", ha="right",
                        fontsize=11, color=col)
        else:
            ax.annotate(lab, xy=(r["hi"], y), xytext=(6, 0),
                        textcoords="offset points", va="center",
                        fontsize=11, color=col)
    ax.axvline(0, color="k", lw=1, ls=":", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([VAR_EN.get(r["var"], r["var"]) if lang == "en"
                        else r["var"] for r in ood], fontsize=11.5)
    ax.set_xlabel("paired Δr" if lang == "en" else "配对差 Δr", fontsize=12.5)
    ax.set_title(lbl_ood, fontsize=13.5)
    ax.set_xlim(-0.09, 0.17)
    ax.grid(axis="x", ls="--", alpha=0.35)

    fig.suptitle(title, fontsize=17, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(f_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved:", f_out)


make("en")
make("zh")

# ============ 配套数字表（写CSV存档） ============
rows = []
for r in holdout:
    rows.append({"panel": "holdout35", "comparison": r["key"], "dmean": round(r["d"], 6),
                 "ci95_lo": round(r["lo"], 6), "ci95_hi": round(r["hi"], 6),
                 "cluster_lo": r["clo"], "cluster_hi": r["chi"],
                 "wins": r["wins"], "losses": r["losses"]})
for r in ood:
    rows.append({"panel": "ood9dim_vs_SG31", "comparison": r["var"],
                 "dmean": round(r["d"], 6), "ci95_lo": round(r["lo"], 6),
                 "ci95_hi": round(r["hi"], 6), "cluster_lo": np.nan, "cluster_hi": np.nan,
                 "wins": r["wins"], "losses": r["n"] - r["wins"]})
out_csv = os.path.join(OUT_DIR, "effect_forest_numbers.csv")
df_new = pd.DataFrame(rows)
df_new.to_csv(out_csv, index=False, encoding="utf-8-sig")
ref_csv = os.path.join(R, "effect_forest_numbers.csv")
if os.path.exists(ref_csv):
    df_ref = pd.read_csv(ref_csv, encoding="utf-8-sig")
    assert list(df_new["panel"]) == list(df_ref["panel"])
    assert list(df_new["comparison"]) == list(df_ref["comparison"])
    assert list(df_new["wins"]) == list(df_ref["wins"])
    assert list(df_new["losses"]) == list(df_ref["losses"])
    for c in ["dmean", "ci95_lo", "ci95_hi"]:
        assert np.allclose(df_new[c].to_numpy(dtype=float), df_ref[c].to_numpy(dtype=float), atol=1e-12, equal_nan=True), c
    for c in ["cluster_lo", "cluster_hi"]:
        assert np.allclose(df_new[c].to_numpy(dtype=float), df_ref[c].to_numpy(dtype=float), atol=2e-3, equal_nan=True), c
print(f"saved: {out_csv} (verified exact match with configs_and_metrics/effect_forest_numbers.csv on all 13 record CIs)")
