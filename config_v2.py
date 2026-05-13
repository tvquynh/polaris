"""P01 v2 — frozen config (top-K per type LOCKED 2026-05-09 from preflight).

Source of truth: preflight_per_type_lofo/results/top_k_per_type.json (n=10 LOGO).

DO NOT mutate at runtime. Override via CLI or experiment driver only.
"""
from __future__ import annotations

# ── File-type experts ────────────────────────────────────────────────────────
PE_FILE_TYPES = ["win32", "win64", "dot_net"]

# ── Top-K per type for MG-EP-PT (LOCKED 2026-05-09) ─────────────────────────
# All FDR-significant where claimed (n=10 Wilcoxon BH-FDR q<0.05).
# Mean ΔTPR_challenge_1pct from preflight 10 seeds × 12 group LOGO.
TOP_K_PER_TYPE = {
    "win32":   ["HDR", "SEC", "IMP"],   # mean ΔTPR +0.0370 / +0.0353 / +0.0217
    "win64":   ["IMP", "DD",  "HDR"],   # mean ΔTPR +0.0259 / +0.0120 / +0.0118
    "dot_net": ["STR", "HDR", "DD"],    # mean ΔTPR +0.0481 / +0.0041 / +0.0011
}

# Top-5 alternative for MG-EP-PT sensitivity ablation (config 5).
TOP_K_PER_TYPE_5 = {
    "win32":   ["HDR", "SEC", "IMP", "AUTH", "DD"],
    "win64":   ["IMP", "DD",  "HDR", "STR",  "SEC"],
    "dot_net": ["STR", "HDR", "DD",  "WARN", "BH"],
}

# ── Drop-bottom-K per type for feature efficiency ablation (config 11a) ────
# Bottom-3 per type (lowest mean ΔTPR; for win64 and dot_net these are negative).
DROP_BOTTOM_3_PER_TYPE = {
    "win32":   ["GFI", "BH",   "EXP"],   # smallest positive ΔTPR
    "win64":   ["WARN", "AUTH", "BH"],   # all negative ΔTPR (drop improves TPR)
    "dot_net": ["BH",  "GFI",  "EXP"],   # all negative ΔTPR
}

# Aggressive drop-bottom-8 for .NET (config 11b — paper-worthy standalone finding).
# All 8 groups have negative or near-zero mean ΔTPR per preflight.
DROP_BOTTOM_8_DOTNET = ["BH", "GFI", "EXP", "AUTH", "IMP", "SEC", "BEH", "RH"]

# ── MG-EP-PT hyperparameters ────────────────────────────────────────────────
MG_EP_BETA = 1.0                       # base scaling (Optuna-tunable)
MG_EP_TOP_K = 3                        # default 3; ablation knob {1, 3, 5}
MG_EP_WEIGHT_MODE = "delta_tpr_weighted"  # {"uniform", "delta_tpr_weighted"}

# Pre-computed alpha_g per type (delta-TPR-weighted; normalized so sum = K).
# Used when MG_EP_WEIGHT_MODE = "delta_tpr_weighted".
MG_EP_ALPHA_PER_TYPE_TOP3 = {
    "win32":   {"HDR": 1.149, "SEC": 1.097, "IMP": 0.674, "_sum": 2.92},
    "win64":   {"IMP": 1.516, "DD":  0.760, "HDR": 0.724, "_sum": 3.00},
    "dot_net": {"STR": 2.768, "HDR": 0.156, "DD":  0.076, "_sum": 3.00},
}

# ── SAR (State-Aware Reweighting) hyperparameters ──────────────────────────
# Note: replaces original CSC (Covariate-Shift Correction) because P11+
# challenge samples are state5=-1 (protected NA per P11+ definition).
# We approximate challenge state distribution by upweighting "evasive analogs":
# state 3 (late_confirmed_malicious) + state 4 (reverted_disputed).
SAR_K_LATE_CONFIRMED = 1.0      # k_3 — Optuna-tunable
SAR_K_REVERTED       = 0.5      # k_4 — Optuna-tunable, smaller (reverted is less malicious)
SAR_STATE_LATE_CONFIRMED = 3
SAR_STATE_REVERTED        = 4

# State labels source path (P11+, full-data run, frozen 2026-05-07)
P11_RELABEL_FRAME_PATH = (
    r"../p11_state5_benchmark/results/state5_benchmark_full/relabel_frame.parquet"
)
P11_STATE_COLUMN = "state5_tau5"  # tau=5 (paper main)
P11_SHA256_COLUMN = "sha256"
P11_FILETYPE_COLUMN = "file_type"

# ── Weight clipping ─────────────────────────────────────────────────────────
WEIGHT_CLIP_R_DEFAULT = 10.0
WEIGHT_CLIP_R_VARIANTS = [5.0, 10.0, 20.0]   # config 12 ablation (R sensitivity)

# ── PETO (Per-Expert Threshold Optimization) ───────────────────────────────
PETO_GRID_STEP = 0.001
PETO_FALLBACK_OPTIMIZER = "SLSQP"
PETO_FPR_BUDGET = 0.01    # ensemble FPR <= 1%
PETO_PRIMARY_FPR = 0.001  # also report at 0.1%

# ── Seeds (project convention 10) ───────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1011, 2026, 3141, 4242, 5555, 6789]
PROTOTYPE_SEEDS = [42, 123]

# ── Train/val split ─────────────────────────────────────────────────────────
VAL_SPLIT = 0.1
SPLIT_SEED = 0  # FIXED across seeds

# ── FPR levels for evaluation ───────────────────────────────────────────────
FPR_LEVELS = [0.001, 0.01]
PRIMARY_METRIC = "tpr_at_fpr_001_challenge"

# ── Baseline-B (Joyce et al. KDD'25 lgbm_config.json) ──────────────────────
BASELINE_B = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "num_leaves": 64,
    "min_data_in_leaf": 100,
    "learning_rate": 0.1,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "feature_fraction": 0.9,
    "feature_fraction_bynode": 0.9,
    "lambda_l1": 0,
    "lambda_l2": 1.0,
    "is_unbalance": True,
    "min_sum_hessian_in_leaf": 0.001,
    "boost_from_average": True,
    "sigmoid": 1.0,
    "max_delta_step": 0,
    "first_metric_only": True,
    "metric": ["auc", "binary_logloss"],
    "verbose": -1,
    "feature_pre_filter": False,
    "device_type": "cpu",
}

# ── Baseline-A (paper-literal — only 3 params from Joyce et al. §4.1 text) ─
BASELINE_A = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "verbose": -1,
    "feature_pre_filter": False,
    "n_estimators": 500,
    "num_leaves": 64,
    "min_data_in_leaf": 100,
}

# Optuna 100-trial best hyperparameters (one-time offline tune on EMBER2024).
# Method: file-type experts with size-aware reweighting (alpha=11, percentile=90).
# Use via CONFIG_REGISTRY config "fte_ls_optuna100" -- replaces in-loop HPO.
BASELINE_OPTUNA_100T = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "num_leaves": 94,
    "min_data_in_leaf": 100,
    "learning_rate": 0.035,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.92,
    "bagging_freq": 1,
    "metric": "binary_logloss",
    "verbose": -1,
    "feature_pre_filter": False,
}
