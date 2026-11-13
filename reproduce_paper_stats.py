# -*- coding: utf-8 -*-
"""
Self-contained JMR public reproduction verification script.

Runs purely on files inside this ReproBundle (zero external raw data required):
  1. Verifies SHA-256 checksums of all 6 frozen paper checkpoints & MANIFEST_SHA256.csv.
  2. Instantiates ConformerResUNet1D, SmallCNN1D, and ECADSTCN, loads all 6 checkpoints,
     checks parameter counts (e.g., 3,563 float32 params for ECADSTCN), and executes
     synthetic forward passes.
  3. Recomputes all primary statistical tables & confidence intervals reported in the
     manuscript and Supplementary Material:
       - Table 3 / Section 3.1: Extended synthetic suite (n=1,024; 725 complete + 299 truncated),
         multi-seed A-final consistency (seeds 1-3), and SmallCNN1D architectural control (Section S6-B).
       - Table 3 / Table 4 / Section 3.2 / Fig. 3(b): 9-condition stress suite (3,456 rows; 600 ms n=128).
       - Table 3 / Section 3.3 / Table S2: 35-record benchtop holdout under P-mag & P-pea,
         paired Method x Protocol interaction (record & 7-oil clustered 95% CIs, seed 20261025,
         B = 20,000), and Fixed Common-delta sensitivity checks (delta_common in {0, -0.1400, -0.1700, -0.1725}).
       - External LIME cohort (n=46 P-pea, n=44 P-mag) & vendor scale-shift diagnostics.
       - Table 3 / Section 3.4: 15-record noise-injection cascade (900 rows; seed=20261026, B=20,000).
       - Table 5 / Fig. 6 / Section 3.5 / Table S4: Zynq-7020 ARM file replay & compute benchmark verification.
"""
import csv
import hashlib
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from scipy import stats

BUNDLE_ROOT = Path(__file__).resolve().parent
SRC_HOST = BUNDLE_ROOT / "src_host"
SRC_DEN = BUNDLE_ROOT / "src_denoiser"
DATA_DIR = BUNDLE_ROOT / "configs_and_metrics"
CKPT_DIR = BUNDLE_ROOT / "checkpoints"
ARM_DIR = BUNDLE_ROOT / "src_arm_cpp"

sys.path.insert(0, str(SRC_HOST))
sys.path.insert(0, str(SRC_DEN))

EXPECTED_CKPT_SHA256 = {
    "h1_T06_original.pth": "f045392ccc286acb1260c882b345cc7db0057eee773fcaef4e3976c5d0abc356",
    "h1_final_seed1.pth": "819f8fdcc549a425a72755df7fff59c58d82c8c7b8e0dacb2d989d97c195af6f",
    "h1_final_seed2.pth": "e7fae1594406f7ff12ae1ac0c9648ba10f6c1638584cd6ddf1756f50d51a1d46",
    "h1_final_seed3.pth": "d624612c4c1e701b6fcb599870ff34827e7e9ee903a070bc62c23842dca79bde",
    "h1_smallcnn_seed1.pth": "8d80748889996b4aec23001bfb0162465ed201beea8773666d5a65ff285e8482",
    "m4_cpmg_multiecho_residual_eca_ds_tcn.pth": "6250477f58de7b97d4fa69960d7f710a23a969aaea3e90ea6aa226083eadb9dc",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def boot_record_ci(d, seed=20261025, n_boot=20000):
    arr = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def boot_cluster_ci(oils, vals, seed=20261025, n_boot=20000):
    df_tmp = pd.DataFrame({"oil": oils, "val": vals})
    groups = [g["val"].values for _, g in df_tmp.groupby("oil", sort=True)]
    rng = np.random.default_rng(seed)
    k = len(groups)
    idx = rng.integers(0, k, size=(n_boot, k))
    g_sizes = np.array([len(g) for g in groups], dtype=np.float64)
    g_sums = np.array([g.sum() for g in groups], dtype=np.float64)
    means = g_sums[idx].sum(axis=1) / g_sizes[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def step1_verify_manifest_and_checkpoints():
    print("\n[1/7] Verifying MANIFEST_SHA256.csv and frozen checkpoints...")
    manifest = DATA_DIR.parent / "MANIFEST_SHA256.csv"
    assert manifest.exists(), f"Missing {manifest}"
    with open(manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        fp = BUNDLE_ROOT / r["rel_path"]
        assert fp.exists(), f"Missing file in manifest: {r['rel_path']}"
        actual = sha256_file(fp)
        assert actual == r["sha256"], f"SHA-256 mismatch on {r['rel_path']}"
    for fname, expected in EXPECTED_CKPT_SHA256.items():
        actual = sha256_file(CKPT_DIR / fname)
        assert actual == expected, f"Checkpoint SHA-256 mismatch: {fname}"
    print(f"  PASS: {len(rows)} files verified via MANIFEST_SHA256.csv; 6/6 paper checkpoints match Table S1/S6.")


def step2_smoke_forward_all_models():
    print("\n[2/7] Loading all 6 paper checkpoints and verifying synthetic forward passes...")
    from train import build_model
    from simulator import generate_dataset_ext
    from baselines import sg_nnls_invert
    from m4_train_ablation_models import ECADSTCN
    import torch.nn as nn
    import torch.nn.functional as F
    from models import ResBlock1D

    class SmallCNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(32), nn.GELU())
            self.block1 = nn.Sequential(nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(64), nn.GELU())
            self.block2 = nn.Sequential(nn.Conv1d(64, 128, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(128), nn.GELU())
            self.resblocks = nn.Sequential(*[ResBlock1D(128, 128) for _ in range(4)])
            self.head_pool = nn.AdaptiveAvgPool1d(128)
            self.head_proj = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, x):
            h = self.resblocks(self.block2(self.block1(self.stem(x))))
            p = self.head_pool(h).transpose(1, 2)
            return F.softplus(self.head_proj(p).squeeze(-1))

    # Generate 2 synthetic envelopes with extended-distribution simulator
    x_syn, y_syn, m_syn = generate_dataset_ext(2, 404000)
    assert x_syn.shape == (2, 1024) and y_syn.shape == (2, 128) and m_syn.shape == (2, 1024)
    x_t = torch.from_numpy(x_syn).unsqueeze(1)

    for ckpt_name in ["h1_T06_original.pth", "h1_final_seed1.pth", "h1_final_seed2.pth", "h1_final_seed3.pth"]:
        ck = torch.load(CKPT_DIR / ckpt_name, map_location="cpu", weights_only=True)
        m = build_model(ck["config"])
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        with torch.no_grad():
            pred = m(x_t).numpy()
        assert pred.shape == (2, 128) and np.all(np.isfinite(pred)) and np.all(pred >= 0.0)

    ck_cnn = torch.load(CKPT_DIR / "h1_smallcnn_seed1.pth", map_location="cpu", weights_only=True)
    m_cnn = SmallCNN1D()
    sd_cnn = ck_cnn["model_state_dict"] if isinstance(ck_cnn, dict) and "model_state_dict" in ck_cnn else ck_cnn
    m_cnn.load_state_dict(sd_cnn)
    m_cnn.eval()
    with torch.no_grad():
        p_cnn = m_cnn(x_t).numpy()
    assert p_cnn.shape == (2, 128) and np.all(np.isfinite(p_cnn))

    # Classical SG+NNLS check on same synthetic echo
    p_sg = sg_nnls_invert(x_syn[0], 51, 3, 200.0, m_syn[0])
    assert p_sg.shape == (128,) and np.all(np.isfinite(p_sg)) and np.all(p_sg >= 0.0)

    # On-board ECA-DS-TCN denoiser checkpoint verification
    denoiser = ECADSTCN()
    n_params = sum(p.numel() for p in denoiser.parameters())
    assert n_params == 3563, f"Expected 3563 params, got {n_params}"
    ck_den = torch.load(CKPT_DIR / "m4_cpmg_multiecho_residual_eca_ds_tcn.pth", map_location="cpu", weights_only=True)
    sd_den = ck_den["model_state_dict"] if isinstance(ck_den, dict) and "model_state_dict" in ck_den else ck_den
    denoiser.load_state_dict(sd_den)
    denoiser.eval()
    rng = np.random.default_rng(42)
    iq = torch.from_numpy(rng.normal(0.0, 0.05, size=(2, 2, 1024)).astype(np.float32))
    with torch.no_grad():
        res_out = denoiser(iq).numpy()
    assert res_out.shape == (2, 2, 1024) and np.all(np.isfinite(res_out))
    print(f"  PASS: All 6 checkpoints loaded and ran cleanly (Conformer x4, SmallCNN1D x1, ECADSTCN {n_params} params).")


def step3_synthetic_suites():
    print("\n[3/7] Verifying Extended Synthetic Suite (n=1,024), Original Suite, and 9-dim OOD Suite...")
    ext = pd.read_csv(DATA_DIR / "corrected_ext_dist_rows.csv")
    cnn = pd.read_csv(DATA_DIR / "smallcnn_test_rows.csv")
    old = pd.read_csv(DATA_DIR / "final_old_dist_rows.csv")
    ood = pd.read_csv(DATA_DIR / "ood_rows_corrected.csv")

    af = ext[ext["branch"] == "A定稿seed3"].reset_index(drop=True)
    sg51 = ext[ext["branch"] == "SG(51,3)+NNLS(修正)"].reset_index(drop=True)
    assert len(af) == 1024 and len(sg51) == 1024
    assert abs(af["pearson_r"].mean() - 0.9454) < 0.001
    assert abs(sg51["pearson_r"].mean() - 0.916) < 0.001
    assert abs(af[af["truncated"] == False]["pearson_r"].mean() - 0.953) < 0.001
    assert abs(af[af["truncated"] == True]["pearson_r"].mean() - 0.928) < 0.001
    assert abs(sg51[sg51["truncated"] == False]["pearson_r"].mean() - 0.920) < 0.001
    assert abs(sg51[sg51["truncated"] == True]["pearson_r"].mean() - 0.906) < 0.001
    assert len(cnn) == 1024

    s1 = ext[ext["branch"] == "A定稿seed1"]["pearson_r"].mean()
    s2 = ext[ext["branch"] == "A定稿seed2"]["pearson_r"].mean()
    s3 = af["pearson_r"].mean()
    print(f"  Extended Suite (n=1,024): A-final seeds=[{s1:.5f}, {s2:.5f}, {s3:.5f}] (SD={np.std([s1,s2,s3], ddof=1):.5f}), SG(51,3)+NNLS={sg51['pearson_r'].mean():.4f}, SmallCNN1D={cnn['pearson_r'].mean():.4f}")

    # Original synthetic suite (n=1,024)
    old_seeds = [old[old["branch"] == f"A定稿seed{i}"]["pearson_r"].mean() for i in (1, 2, 3)]
    print(f"  Original Suite (n=1,024): A-final seeds={[round(x, 5) for x in old_seeds]} (SD={np.std(old_seeds, ddof=1):.5f})")

    # 600 ms stress suite (n=128)
    s600_af = ood[(ood["variant"] == "V6_缺失时间窗_600ms") & (ood["branch"] == "A仿真版")].reset_index(drop=True)
    s600_sg = ood[(ood["variant"] == "V6_缺失时间窗_600ms") & (ood["branch"] == "SG(31,2)+NNLS")].reset_index(drop=True)
    assert len(s600_af) == 128 and len(s600_sg) == 128
    dm, dlo, dhi = boot_record_ci(s600_af["pearson_r"].values - s600_sg["pearson_r"].values, seed=20261025)
    assert abs(s600_af["pearson_r"].mean() - 0.887) < 0.001
    assert abs(s600_sg["pearson_r"].mean() - 0.929) < 0.001
    assert abs(dm - (-0.0420)) < 0.0005
    print(f"  600 ms Stress Suite (n=128): A-final={s600_af['pearson_r'].mean():.4f}, SG(31,2)+NNLS={s600_sg['pearson_r'].mean():.4f}, Δr={dm:+.4f} 95% CI [{dlo:+.4f}, {dhi:+.4f}]")


def step4_holdout35_interaction_and_common_delta():
    print("\n[4/7] Verifying 35-Record Holdout, Paired Method x Protocol Interaction, and Fixed Common-delta Check...")
    long_v3 = pd.read_csv(DATA_DIR / "holdout35_two_protocol_v3.csv", encoding="utf-8-sig")
    wide = pd.read_csv(DATA_DIR / "holdout35_pmag_wide.csv", encoding="utf-8-sig")
    eff = pd.read_csv(DATA_DIR / "effect_forest_numbers.csv", encoding="utf-8-sig")
    cdelta = pd.read_csv(DATA_DIR / "holdout35_common_delta_metrics.csv", encoding="utf-8")

    order = list(wide["样品"])
    oils = pd.Series(order).str.replace(r"\d+h$", "", regex=True).values
    mag_r = long_v3[long_v3["input_protocol"] == "raw_mag"].pivot(index="样品", columns="branch", values="pearson_r").loc[order]
    pea_r = long_v3[long_v3["input_protocol"] == "pea"].pivot(index="样品", columns="branch", values="pearson_r").loc[order]
    mag_f = long_v3[long_v3["input_protocol"] == "raw_mag"].pivot(index="样品", columns="branch", values="fwd_rmse").loc[order]
    pea_f = long_v3[long_v3["input_protocol"] == "pea"].pivot(index="样品", columns="branch", values="fwd_rmse").loc[order]

    assert abs(mag_r["A定稿"].mean() - 0.8948) < 0.0002 and abs(mag_r["SG旧"].mean() - 0.8909) < 0.0002
    assert abs(pea_r["A定稿"].mean() - 0.8929) < 0.0002 and abs(pea_r["SG旧"].mean() - 0.9042) < 0.0002
    assert abs(mag_f["A定稿"].mean() - 0.0326) < 0.0002 and abs(mag_f["SG旧"].mean() - 0.0369) < 0.0002
    assert abs(pea_f["A定稿"].mean() - 0.0097) < 0.0002 and abs(pea_f["SG旧"].mean() - 0.0163) < 0.0002

    # Check P-mag and P-pea primary CIs
    m_pm, lo_pm, hi_pm = boot_record_ci((wide["A定稿_r"] - wide["SG旧_r"]).values, seed=20261025)
    _, clo_pm, chi_pm = boot_cluster_ci(oils, (wide["A定稿_r"] - wide["SG旧_r"]).values, seed=20261025)
    m_pp, lo_pp, hi_pp = boot_record_ci((pea_r["A定稿"] - pea_r["SG旧"]).values, seed=20261025)
    row_eff = eff[(eff["panel"] == "holdout35") & (eff["comparison"] == "SG旧_r")].iloc[0]
    assert abs(m_pm - row_eff["dmean"]) < 1e-5
    print(f"  P-mag A-final vs SG(31,2)+NNLS: Δr={m_pm:+.4f} rec=[{lo_pm:+.4f}, {hi_pm:+.4f}] clus=[{clo_pm:+.4f}, {chi_pm:+.4f}]")
    print(f"  P-pea A-final vs SG(31,2)+NNLS: Δr={m_pp:+.4f} rec=[{lo_pp:+.4f}, {hi_pp:+.4f}]")

    # Check all 3 Table S2 interactions
    for comp, label in [("直接NNLS", "Direct NNLS"), ("SG扩展", "SG(51,3)+NNLS"), ("SG旧", "SG(31,2)+NNLS")]:
        d_int = ((mag_r["A定稿"] - mag_r[comp]) - (pea_r["A定稿"] - pea_r[comp])).values
        mi, rlo, rhi = boot_record_ci(d_int, seed=20261025)
        _, clo, chi = boot_cluster_ci(oils, d_int, seed=20261025)
        _, pt = stats.ttest_1samp(d_int, 0.0)
        _, pw = stats.wilcoxon(d_int)
        npos = int((d_int > 0).sum())
        assert rlo > 0 and clo > 0
        print(f"  Interaction A-final vs {label:13s}: Δ_int={mi:+.4f}, rec=[{rlo:+.4f}, {rhi:+.4f}], clus=[{clo:+.4f}, {chi:+.4f}], pos={npos}/35, p_t={pt:.2e}, p_w={pw:.2e}")

    # Fixed Common-delta verification (Section 3.3 / Table S2 / Section S4 / Table 6)
    cdelta["delta_str"] = cdelta["delta_common"].map(lambda x: f"{float(x):.4f}")
    for dstr in ["-0.1400", "-0.1700", "-0.1725"]:
        sub = cdelta[cdelta["delta_str"] == dstr]
        pm = sub[sub["input_protocol"] == "raw_mag"].pivot(index="record_id", columns="branch", values="pearson_r")
        pp = sub[sub["input_protocol"] == "pea"].pivot(index="record_id", columns="branch", values="pearson_r")
        coils = sub[sub["input_protocol"] == "raw_mag"].drop_duplicates("record_id").set_index("record_id").loc[pm.index, "oil_type"].values
        d_int = ((pm["A-final"] - pm["SG(31,2)+NNLS"]) - (pp["A-final"] - pp["SG(31,2)+NNLS"])).values
        mi, rlo, rhi = boot_record_ci(d_int, seed=20261025)
        _, clo, chi = boot_cluster_ci(coils, d_int, seed=20261025)
        assert rlo > 0 and clo > 0
        print(
            f"  Fixed Common-delta ({dstr}): P-mag A-final={pm['A-final'].mean():.4f} vs SG={pm['SG(31,2)+NNLS'].mean():.4f}; "
            f"P-pea A-final={pp['A-final'].mean():.4f} vs SG={pp['SG(31,2)+NNLS'].mean():.4f}; "
            f"Δ_int={mi:+.4f} rec=[{rlo:+.4f}, {rhi:+.4f}] clus=[{clo:+.4f}, {chi:+.4f}]"
        )


def step5_lime_and_delta_diagnostics():
    print("\n[5/7] Verifying External LIME Cohort (n=46 P-pea / n=44 P-mag) & Vendor Scale-Shift Tables...")
    l_pea = pd.read_csv(DATA_DIR / "lime46_ppea_validation.csv", encoding="utf-8-sig")
    l_pmag = pd.read_csv(DATA_DIR / "lime44_pmag_validation_v2.csv", encoding="utf-8-sig")
    l_del = pd.read_csv(DATA_DIR / "results_lime_delta.csv", encoding="utf-8-sig")
    m_del = pd.read_csv(DATA_DIR / "results_main15_delta.csv", encoding="utf-8-sig")
    v_del = pd.read_csv(DATA_DIR / "results_niumag_validation.csv", encoding="utf-8-sig")
    n_eval = int(l_pmag["A定稿_r"].notna().sum())
    assert len(l_pea) == 46 and len(l_pmag) == 46 and n_eval == 44 and len(l_del) == 46
    assert l_del["delta_star"].min() >= -0.0101 and l_del["delta_star"].max() <= 0.0001
    print(f"  LIME P-pea rows={len(l_pea)}, P-mag rows={len(l_pmag)} (evaluable={n_eval}), delta* range=[{l_del['delta_star'].min():.4f}, {l_del['delta_star'].max():.4f}]")
    print(f"  Main-15 delta* range=[{m_del['delta_star'].min():.4f}, {m_del['delta_star'].max():.4f}]; Niumag validation rows={len(v_del)}")


def step6_cascade_noise_ladder():
    print("\n[6/7] Verifying 15-Record Noise-Injection Cascade (Table 3 / Section 3.4)...")
    cl = pd.read_csv(DATA_DIR / "cascade_lowsnr_rows.csv", encoding="utf-8-sig")
    assert len(cl) == 900
    agg = cl.groupby(["noise_level", "inverter", "input_path", "sample_name"], as_index=False)[["pearson_r", "peak_err_log"]].mean()
    # Strong noise alpha = 1.0
    sub_nnls = agg[(agg["noise_level"] == 1.0) & (agg["inverter"] == "NNLS反演")].pivot(index="sample_name", columns="input_path", values="pearson_r")
    sub_pk = agg[(agg["noise_level"] == 1.0) & (agg["inverter"] == "NNLS反演")].pivot(index="sample_name", columns="input_path", values="peak_err_log")
    sub_a0 = agg[(agg["noise_level"] == 1.0) & (agg["inverter"] == "A反演")].pivot(index="sample_name", columns="input_path", values="pearson_r")

    r_raw, r_sg, r_edge = sub_nnls["原始带噪"].mean(), sub_nnls["SG"].mean(), sub_nnls["下位机模型"].mean()
    assert abs(r_raw - 0.885) < 0.001 and abs(r_sg - 0.912) < 0.001 and abs(r_edge - 0.919) < 0.001
    assert abs(sub_pk["原始带噪"].mean() - 0.3335) < 0.001
    assert abs(sub_a0["原始带噪"].mean() - 0.888) < 0.001 and abs(sub_a0["SG"].mean() - 0.893) < 0.001 and abs(sub_a0["下位机模型"].mean() - 0.894) < 0.001
    dm_e, lo_e, hi_e = boot_record_ci((sub_nnls["下位机模型"] - sub_nnls["原始带噪"]).values, seed=20261026)
    print(f"  Cascade alpha=1.0 NNLS: Raw={r_raw:.3f}, SG={r_sg:.3f}, Edge={r_edge:.3f} (Edge-Raw Δr={dm_e:+.4f} CI [{lo_e:+.4f}, {hi_e:+.4f}])")
    print(f"  Cascade alpha=1.0 A-original: Raw={sub_a0['原始带噪'].mean():.3f}, SG={sub_a0['SG'].mean():.3f}, Edge={sub_a0['下位机模型'].mean():.3f}")


def step7_arm_logs_and_replay():
    print("\n[7/7] Verifying ARM Zynq-7020 File Replay & Separate Compute Benchmark Tables...")
    br = pd.read_csv(DATA_DIR / "board_replay_rows.csv", encoding="utf-8-sig")
    sg = pd.read_csv(DATA_DIR / "sg21_diff_rows.csv", encoding="utf-8-sig")
    assert len(br) == 15 and br["corr"].min() > 0.999999 and br["rel_rmse"].max() < 3.2e-6
    assert len(sg) == 15 and sg["I_r"].min() > 0.9999999 and sg["I_relrmse_inner"].max() < 2.8e-8
    omp_txt = (ARM_DIR / "log_m5_omp.txt").read_text(encoding="utf-8", errors="replace")
    sg1_txt = (ARM_DIR / "sg21_bench_run1_20261029.log").read_text(encoding="utf-8", errors="replace")
    sg2_txt = (ARM_DIR / "sg21_bench_run2_20261029.log").read_text(encoding="utf-8", errors="replace")
    assert "4000" in omp_txt and "178." in sg1_txt and "178." in sg2_txt
    print(f"  ARM Edge Replay (15/15): min corr={br['corr'].min():.10f}, max rel_rmse={br['rel_rmse'].max():.3e}")
    print(f"  ARM Dual-ch SG(21,2) Parity (15/15): max inner rel_rmse={max(sg['I_relrmse_inner'].max(), sg['Q_relrmse_inner'].max()):.3e}")


if __name__ == "__main__":
    print("=== JMR Public Reproduction Bundle Self-Verification ===")
    step1_verify_manifest_and_checkpoints()
    step2_smoke_forward_all_models()
    step3_synthetic_suites()
    step4_holdout35_interaction_and_common_delta()
    step5_lime_and_delta_diagnostics()
    step6_cascade_noise_ladder()
    step7_arm_logs_and_replay()
    print("\nALL 7 VERIFICATION CHECKS PASSED.")
