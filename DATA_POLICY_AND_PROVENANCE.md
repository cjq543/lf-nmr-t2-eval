# Data Policy and Asset Provenance Statement

1. **Publicly Redistributed Code, Weights, and Derived Scalar Metrics**:
   All synthetic data simulation scripts, neural/classical model implementations, C++/NEON on-board replay sources, frozen weight checkpoints, hardware execution logs, and per-record derived evaluation CSV tables in this bundle contain zero proprietary experimental waveforms and zero unapproved third-party raw workbooks.
2. **Restricted Raw Experimental Recordings**:
   - **Benchtop Niumag 21 MHz CPMG records (`15` main cohort + `35` historical holdout + `93` sealed reserve records)**: Measured by laboratory co-researchers (including senior co-worker / group archive ledgers); because raw `.Fid`/`.pea`/`.Inv` signal files have not been independently released as open raw data, they are excluded from public redistribution. Derived per-record evaluation metrics are included in `configs_and_metrics/`. Access to raw recordings may be requested from the corresponding author pending institutional and data-owner consent.
   - **External LIME 22 MHz portable-analyzer cohort (`46` records)**: Originated from Ref. [5] (Liu et al.) and withheld in raw form due to third-party ownership restrictions. Derived metric tables (`lime46_ppea_validation.csv`, `lime44_pmag_validation_v2.csv`, `results_lime_delta.csv`) are included.
3. **Standalone Synthetic Reproducibility**:
   Synthetic suites in `src_host/simulator.py` and `src_denoiser/m4_generate_cpmg_multiecho_dataset.py` are completely deterministic from fixed seeds (`seed = 42`, `303000`, `404000`) and run offline without any external dataset dependency.
