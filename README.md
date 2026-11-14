# LF-NMR T₂ Inversion & Echo-Processing Pipeline — Public Reproduction Bundle

This self-contained reproduction bundle accompanies the manuscript:

> **Comparative Evaluation of Low-Field NMR T₂ Inversion across Observation Windows and Echo-Envelope Protocols**
>
> Jie Huang, Hao Wei, Di Wang, Caidie Qian, Dingwen Zhou, Can Liang, Zijian Jia* (*Corresponding author: jiazijian1989@usst.edu.cn). Submitted to *Journal of Magnetic Resonance*.
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
3. **`src_arm_cpp/`**: ARM Cortex-A9 Zynq-7020 C++/OpenMP headers, single-precision `float32` weight export header (`m4_ds_cnn_weights.hpp`), streaming file-replay source (`replay_main.cpp`), dual-channel SG(21,2) timing/boundary verification C++ benchmarks, compiler flags, and raw execution logs (`log_m5_omp.txt`, `log_m5_single.txt`, `replay_timing.log`, `sg21_bench_run*.log`).
4. **`configs_and_metrics/`**: Frozen JSON solver configurations (`classical_params*.json`), host CPU fixed-input timing JSON (`host_pc_benchmark_checks.json`), and all **derived per-record scalar metric CSV tables** used for Tables 2–6, Figs. 2–6, and Supplementary Tables S2–S4 (`pearson_r`, `peak_err_log`, `spec_l2`, `fwd_rmse`, `rel_rmse`, bootstrap confidence intervals, and the derived `Fixed Common-δ` sensitivity metric table `holdout35_common_delta_metrics.csv`).
5. **`scripts_figures_and_stats/`**: Standalone publication figure, Graphical Abstract (`2.5:1` / `300 dpi`), and bootstrap recomputation scripts (`generate_en_figures.py`, `generate_effect_forest.py`, `pea_ci_recompute_20261029.py`, etc.), runnable offline using only bundle-local files with outputs written under `results/regenerated_figures/` and `results/`.

### Strictly Excluded (Raw Experimental Waveforms & Third-Party Files)
No raw experimental waveforms, proprietary instrument files, compiled executables, or unused checkpoints appear in this bundle:
- **Benchtop Niumag CPMG raw recordings (`15 + 35` records and `93` sealed reserve records)** (`.Fid`, `.pea`, `.Inv` files) originate from institutional/laboratory-acquired group datasets (`Liu et al.` / group archive) without independent public raw-data release.
- **External portable-analyzer LIME raw cohort (`46` records)** (`.org`, `_data.xlsx` reference workbooks) belong to third-party published literature (Ref. [5] of Main Text) without redistribution rights.
- **Intermediate binary replay waveforms** (`*.iq.bin`, `*.t.bin`, `*.tar.gz`, `*.npy`, large `*.npz` caches) and unused `h2_*` exploratory checkpoints are excluded.

All derived scalar tables needed to recompute every paper table, statistical comparison, paired `Method × Protocol` interaction interval, and sensitivity check are provided in `configs_and_metrics/`. Raw waveforms/workbooks may be requested through the corresponding author subject to data-owner / institutional approval (or directly from the authors of Ref. [5] for LIME workbooks).

---

## 2. Quickstart Verification (One Command)

Requires Python 3.10–3.12 with packages listed in `requirements.txt` (`numpy`, `pandas`, `scipy`, `torch`, `matplotlib`, `pillow`; CPU is sufficient). Install dependencies and run from the bundle root:

```bash
pip install -r requirements.txt
python -B reproduce_paper_stats.py
```

Optional standalone figure & bootstrap scripts (write derived PNG/TIFF/CSV outputs into `results/` without overwriting tracked `MANIFEST_SHA256.csv` inputs):

```bash
python -B scripts_figures_and_stats/generate_en_figures.py
python -B scripts_figures_and_stats/generate_effect_forest.py
python -B scripts_figures_and_stats/pea_ci_recompute_20261029.py
```

`reproduce_paper_stats.py` automatically verifies:
1. Full `MANIFEST_SHA256.csv` integrity and all 6 checkpoint SHA-256 fingerprints;
2. Live loading and synthetic forward inference of `A-original`, `A-final` (seeds 1–3), `SmallCNN1D`, `SG+NNLS`, and `ECADSTCN` (`3,563` float32 params);
3. Exact numerical reproduction of Table 3 (Extended & Original synthetic suites, 35-record `P-mag` vs. `P-pea` holdout, 15-record cascade noise ladder, and ARM hardware consistency), Table 4 / Fig. 3(b) (`600 ms` severe-cutoff & 9-dim perturbation stress suite), Table S2 / Section S4 (`P-mag` vs. `P-pea`, `Method × Protocol` interaction `Δ_int = +0.0152`, and `Fixed Common-δ` sensitivity checks), External LIME & scale-shift diagnostics, and Table 5 / Fig. 6 (`Zynq-7020` ARM replay parity & timing logs).
   - *Note on Fixed Common-δ bootstrap order*: Because seeded percentile bootstrap (`seed = 20261025`, `20,000` resamples) depends slightly (`~0.0002` at the fourth decimal place) on record ordering, `reproduce_paper_stats.py` explicitly aligns `holdout35_common_delta_metrics.csv` via `sample_label_cn` to the canonical sample order of `holdout35_pmag_wide.csv` (`order = list(wide["样品"])`) so all Section S4 confidence intervals reproduce to the fourth decimal place.

---

## 3. Filename Alias Mapping (Archived Internal Names → Public Bundle Paths)

Supplementary Sections S1, S2, S5, S6, and S9 cite archived internal workspace paths (retained for provenance auditing). Their self-contained public equivalents in this bundle are mapped below:

| Archived Internal Path / Alias (Manuscript & Supplement) | Public Bundle Path (`lf-nmr-t2-eval`) |
|---|---|
| `h1_T06_退火长训.pth` | `checkpoints/h1_T06_original.pth` |
| `02_冻结结果权重与代码/models/h1_final_seed{1,2,3}.pth`, `h1_smallcnn_seed1.pth` | `checkpoints/h1_final_seed{1,2,3}.pth`, `checkpoints/h1_smallcnn_seed1.pth` |
| `02_冻结结果权重与代码/results/classical_params*.json` | `configs_and_metrics/classical_params*.json` |
| `02_冻结结果权重与代码/results/final_{ext_dist,old_dist,real15}_rows.csv` | `configs_and_metrics/final_{ext_dist,old_dist,real15}_rows.csv` |
| `04_审阅核验链/11_基线修正_20261025/corrected_ext_dist_rows.csv` | `configs_and_metrics/corrected_ext_dist_rows.csv` |
| `04_审阅核验链/13_现代基线对照_20261025/smallcnn_test_rows.csv` | `configs_and_metrics/smallcnn_test_rows.csv` |
| `02_冻结结果权重与代码/results/ood_rows*.csv`, `sim_lowsnr_rows.csv` | `configs_and_metrics/ood_rows*.csv`, `sim_lowsnr_rows.csv` |
| `35条冻结留出验证结果.csv` (`19_35条留出结果复核_20261025/`) | `configs_and_metrics/holdout35_pmag_wide.csv` |
| `35条冻结模型_协议修正_v3.csv` (`21_协议修正再分析_20261026/`) | `configs_and_metrics/holdout35_two_protocol_v3.csv` |
| `holdout35_common_delta_metrics.csv` (`04_审阅核验链/23_.../`) | `configs_and_metrics/holdout35_common_delta_metrics.csv` |
| `03_外部数据资产_46条LIME/外部验证全量表.csv` | `configs_and_metrics/lime46_ppea_validation.csv` |
| `03_外部数据资产_46条LIME/原始级外部验证全量表_v2.csv` | `configs_and_metrics/lime44_pmag_validation_v2.csv` |
| `03_外部数据资产_46条LIME/07_原始级别外部验证_v2.py` | `src_host/07_raw_lime_validation_v2.py` |
| `02_冻结结果权重与代码/results_新增/pea_ci_recompute_20261029.{py,txt}` | `scripts_figures_and_stats/pea_ci_recompute_20261029.py`, `docs/pea_ci_recompute_20261029.txt` |
| `01_论文正文/effect_forest_numbers.csv`, `图表脚本/generate_{effect_forest,en_figures}.py` | `configs_and_metrics/effect_forest_numbers.csv`, `scripts_figures_and_stats/generate_{effect_forest,en_figures}.py` |
| `04_审阅核验链/04_审阅与证据链/review_20261024/additional_checks.json` | `configs_and_metrics/host_pc_benchmark_checks.json` |
| `08_板端证据/board_benchmark_real_20261024/log_*.txt` | `src_arm_cpp/log_*.txt` |
| `08_板端证据/板端回放_20261024/data_down/replay_timing.log`, `pc_ref_summary.json` | `src_arm_cpp/replay_timing.log`, `configs_and_metrics/pc_ref_summary.json` |
| `02_冻结结果权重与代码/results_新增/sg21_matched_20261029/` (`sg21_diff_rows.csv`, `*.cpp`, `*.log`, `_verify_edge_fix.py`) | `configs_and_metrics/sg21_diff_rows.csv`, `src_arm_cpp/` |
| `_jmr_edit_20261106/content_format_review/denoiser_sources/{m4_*.py,*.pth}` | `src_denoiser/m4_*.py`, `checkpoints/m4_cpmg_multiecho_residual_eca_ds_tcn.pth` |

*Author-retained internal assets (not redistributed in public bundle)*: `05_数据资产总账与清单/` (`数据资产总账.md`, `数据指针清单.md`), `16_总库/` (signal collision audit CSV), `13_文献核对_20261027/` (bibliographic verification dossier), raw Niumag `.Fid`/`.pea`/`.Inv` files, third-party Ref. [5] LIME `.xlsx`/`.org` workbooks, and the cached `denoiser_data/m4_cpmg_multiecho_dataset.npz` archive (which is deterministically regenerable offline via `src_denoiser/m4_generate_cpmg_multiecho_dataset.py` with `seed = 42`).

---

## 4. Historical Code Comments & Study-Design Terminology

Several archived scripts in `src_host/` (`evaluate.py`, `hconfig.py`, `run_all.py`, `train.py`, `cascade_experiment.py`, `realdata.py`, `b2_final_eval.py`) preserve their original development headers and comments mentioning `Test3`, `盲测` (blind test), `Train9/Val3/Test3`, or exploratory fine-tuning branches (`m6`). Those strings document historical development provenance only. As explicitly stated in **Main Text Sections 2.2, 2.4, and 4.2** and **Supplementary Section S7**, all measured evaluations reported in the paper (`35`-record benchtop holdout, `15`-record cascade replay, and `46`-record LIME) are **retrospective / post-hoc or protocol-corrected evaluations under frozen checkpoints** rather than prospective blind clinical/field trials; the `93`-record prospective reserve cohort remains sealed and completely excluded from this codebase and from model development.
