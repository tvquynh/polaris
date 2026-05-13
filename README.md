# Comprehensive Ablation of Reweighting Strategies for Evasive PE Malware Detection

Reproducibility artifact for the manuscript *"Comprehensive Ablation of
Reweighting Strategies for Detecting Evasive PE Malware on EMBER2024:
Performance, Compute, and Adversarial Robustness Trade-offs"* (under review,
Engineering Applications of Artificial Intelligence).

## Headline result

- **Method:** File-Type Experts + Size-aware reweighting + Optuna-100-trial tuned hyperparameters
- **Performance:** TPR @ 1% FPR = **76.49% ± 0.24%** on EMBER2024 challenge set
- **Improvement:** **+16.33 percentage points** over Joyce et al. KDD'25 LightGBM baseline
- **Stats:** n = 10 seeds, paired Wilcoxon p = 0.002, Cohen's d = 12.80, BH-FDR q = 0.0021

## Repository layout

```
.
├── pipeline_v2.py                       Main runner — train + eval per configuration
├── config_v2.py                         Frozen hyperparameters + per-type top-K
├── multi_group_ep_per_type.py           Multi-group entropy-proximity reweighting
├── state_aware_reweighting.py           State-aware reweighting using P11+ states
├── peto.py                              Per-expert threshold optimization
├── ember_v3_schema.py                   EMBER2024 v3 feature schema (12 groups)
├── ember2024_parquet_loader.py          Vendored parquet loader
├── aggregate_phase4.py                  Master statistical aggregation
├── generate_figures.py                  Manuscript figures (matplotlib)
├── feature_importance_analysis.py       Feature importance dispersion analysis
├── perturbation_robustness.py           Synthetic perturbation robustness test
├── preflight/                           Per-type LOGO preflight (input for MG-EP-PT)
│   ├── feature_groups.py
│   ├── aggregate_per_type.py
│   ├── run_per_type_logo.py
│   └── config_preflight.py
├── scripts/                             Reproduction shell scripts
│   ├── run_phase4a_ablation.sh          No-HPO ablation (15 configs × 10 seeds)
│   └── run_phase4b_optuna100.sh         Winner config × 10 seeds with fixed Optuna params
├── tests/                               Unit tests (pytest)
│   ├── test_v2_modules.py
│   ├── test_sar_peto.py
│   └── test_pipeline_v2.py
├── LICENSE
├── requirements.txt
└── README.md
```

## Environment setup

```bash
python -m venv venv
source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Tested with Python 3.10.12.

## Data

This work uses the publicly available
[EMBER2024 dataset](https://github.com/FutureComputing4AI/EMBER2024).
Download the parquet files (`ember2024_train.parquet`, `ember2024_test.parquet`,
`ember2024_challenge.parquet`) and place them in a single directory.

State-aware reweighting (SAR) additionally uses a relabel frame derived from
temporally re-checked VirusTotal labels. The relabel frame is not included
in this repository; instructions for reproducing it from the EMBER2024
release plus a VirusTotal API key are provided in
`scripts/derive_relabel_frame.md` (see manuscript Section 5).

## Quick start

```bash
# Single-seed test on baseline-B configuration
python pipeline_v2.py \
    --config_id baseline_B --seed 42 \
    --parquet_dir /path/to/ember2024_parquet \
    --relabel_frame /path/to/relabel_frame.parquet \
    --output_dir ./results \
    --num_threads 60 \
    --no_hpo

# Phase 4a (no-HPO ablation across 15 configurations × 10 seeds)
PYTHON_BIN=$(which python) PARQUET_DIR=/path/to/ember2024 \
    RELABEL_FRAME=/path/to/relabel.parquet \
    OUTPUT_DIR=./results \
    bash scripts/run_phase4a_ablation.sh

# Phase 4b (winner configuration with 100-trial Optuna params × 10 seeds)
bash scripts/run_phase4b_optuna100.sh

# Aggregate results + generate figures
python aggregate_phase4.py --results_dir ./results --out_dir ./results/aggregated
python generate_figures.py --results_dir ./results --out_dir ./results/aggregated/figures

# Feature importance + adversarial robustness analyses (requires --save_model retrain)
python feature_importance_analysis.py \
    --models_dir ./results_models --out_dir ./analysis
python perturbation_robustness.py \
    --models_dir ./results_models --parquet_dir /path/to/ember2024 \
    --out_dir ./analysis
```

## Citation

```
@article{trinh2026comprehensive,
  author  = {Van-Quynh Trinh and Trong-Thua Huynh and De-Thu Huynh and Ngoc-Hieu Le},
  title   = {Comprehensive Ablation of Reweighting Strategies for Detecting
             Evasive PE Malware on EMBER2024: Performance, Compute, and
             Adversarial Robustness Trade-offs},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Contact

Van-Quynh Trinh (Posts and Telecommunications Institute of Technology, Hanoi):
`quynhtv@ptit.edu.vn`
