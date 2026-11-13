# LF-NMR T₂ Inversion & Echo-Processing Pipeline — Public Reproduction Bundle

This self-contained reproduction bundle accompanies the manuscript:

> **Comparative Evaluation of Low-Field NMR T₂ Inversion across Observation Windows and Echo-Envelope Protocols**
>
> Jie Huang, Hao Wei, Di Wang, Can Liang, Caidie Qian, Dingwen Zhou, Zijian Jia* (*Corresponding author: jiazijian1989@usst.edu.cn). Submitted to *Journal of Magnetic Resonance*.
>
> Repository: https://github.com/cjq543/lf-nmr-t2-eval

---

## 1. What Is Included vs. Excluded (Data Governance & Privacy Scope)

### Included in This Public Bundle
1. **`checkpoints/`**: All 6 frozen neural checkpoints cited in Supplementary Tables S1 & S6 (stored under portable ASCII filenames; byte-identical SHA-256 checksums verified in `MANIFEST_SHA256.csv`):
   - `h1_T06_original.pth` *(archived bilingual alias: `h1_T06_退火长训.pth`)* — **A-original** (`1D-Conformer-ResUNet`, original synthetic distribution)
   - `h1_final_seed1.pth`, `h1_final_seed2.pth`, `h1_final_seed3.pth` — **A-final** (`1D-Conformer-ResUNet`, extended distribution + random truncation; `seed 3` is the pre-selected primary checkpoint)
   - `h1_smallcnn_seed1.pth` — **SmallCNN1D** matched-budget architectural control (Section S6-B)
   - `m4_cpmg_multiecho_residual_eca_ds_tcn.pth` — **ECA-DS-TCN** on-board residual denoiser (`3,563` float32 params, `14.25 kB`; Table S5)
2. **`src_host/` & `src_denoiser/`**: Complete synthetic NMR forward/noise/truncation generators, neural architectures (`ConformerResUNet1D`, `SmallCNN1D`, `ECADSTCN`), training & evaluation scripts, and classical Savitzky–Golay + Tikhonov-NNLS solvers.
3. **`src_arm_cpp/`**: ARM Cortex-A9 Zynq-7020 C++/OpenMP headers, quantized/float32 weight export header (`m4_ds_cnn_weights.hpp`), streaming file-replay source (`replay_main.cpp`), dual-channel SG(21,2) timing/boundary verification C++ benchmarks, compiler flags, and raw execution logs (`log_m5_omp.txt`, `log_m5_single.txt`, `sg21_bench_run*.log`).
4. **`configs_and_metrics/`**: Frozen JSON solver configurations (`classical_params*.json`) and all **derived per-record scalar metric CSV tables** used for Tables 2–6, Figs. 2–6, and Supplementary Tables S2–S4 (`pearson_r`, `peak_err_log`, `spec_l2`, `fwd_rmse`, `rel_rmse`, bootstrap confidence intervals, and the derived `Fixed Common-δ` sensitivity metric table `holdout35_common_delta_metrics.csv`).
5. **`scripts_figures_and_stats/`**: Publication figure and bootstrap scripts.

### Strictly Excluded (Raw Experimental Waveforms & Third-Party Files)
No raw experimental waveforms, proprietary instrument files, compiled executables, or unused checkpoints appear in this bundle:
- **Benchtop Niumag CPMG raw recordings (`15 + 35` records and `93` sealed reserve records)** (`.Fid`, `.pea`, `.Inv` files) originate from institutional/laboratory-acquired group datasets (`Liu et al.` / group archive) without independent public raw-data release.
- **External portable-analyzer LIME raw cohort (`46` records)** (`.org`, `_data.xlsx` reference workbooks) belong to third-party published literature (Ref. [5] of Main Text) without redistribution rights.
- **Intermediate binary replay waveforms** (`*.iq.bin`, `*.t.bin`, `*.tar.gz`, `*.npy`, large `*.npz` caches) and unused `h2_*` exploratory checkpoints are excluded.

All derived scalar tables needed to recompute every paper table, statistical comparison, paired `Method × Protocol` interaction interval, and sensitivity check are provided in `configs_and_metrics/`. Raw waveforms/workbooks may be requested through the corresponding author subject to data-owner / institutional approval (or directly from the authors of Ref. [5] for LIME workbooks).

---

## 2. Quickstart Verification (One Command)

Requires Python 3.10–3.12 with `numpy`, `pandas`, `scipy`, `torch` (CPU is sufficient). Run from the bundle root:

```bash
python -B reproduce_paper_stats.py
```

`reproduce_paper_stats.py` automatically verifies:
1. Full `MANIFEST_SHA256.csv` integrity and all 6 checkpoint SHA-256 fingerprints;
2. Live loading and synthetic forward inference of `A-original`, `A-final` (seeds 1–3), `SmallCNN1D`, `SG+NNLS`, and `ECADSTCN` (`3,563` float32 params);
3. Exact numerical reproduction of Table 2 (Extended & Original synthetic suites), Table 3 / Fig. 3(b) (`600 ms` severe-cutoff suite & 9-dim OOD), Table 4 / Table S2 / Section S4 (`P-mag` vs. `P-pea`, `Method × Protocol` interaction `Δ_int = +0.0152`, and `Fixed Common-δ` sensitivity checks), External LIME & scale-shift diagnostics, Table 5 (`15-record` cascade noise ladder), and Fig. 6 (`Zynq-7020` ARM replay parity & timing logs).

---

## 3. Filename Alias Mapping (Archived Bilingual Names → Bundle Names)

| Manuscript / Supplement Cited Alias | Clean Bundle Path |
|---|---|
| `h1_T06_退火长训.pth` | `checkpoints/h1_T06_original.pth` |
| `35条冻结留出验证结果.csv` | `configs_and_metrics/holdout35_pmag_wide.csv` |
| `35条冻结模型_协议修正_v3.csv` | `configs_and_metrics/holdout35_two_protocol_v3.csv` |
| `03_外部数据资产_46条LIME/外部验证全量表.csv` | `configs_and_metrics/lime46_ppea_validation.csv` |
| `03_外部数据资产_46条LIME/原始级外部验证全量表_v2.csv` | `configs_and_metrics/lime44_pmag_validation_v2.csv` |
| `03_外部数据资产_46条LIME/07_原始级别外部验证_v2.py` | `src_host/07_raw_lime_validation_v2.py` |
