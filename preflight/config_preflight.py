"""Frozen hyperparameters for preflight per-type LOGO.

Mirrors P0Y/P01 conventions. Vendored constants — DO NOT diverge from
Joyce et al. KDD'25 lgbm_config.json (Baseline-B).
"""
from __future__ import annotations

# LightGBM Baseline-B (Joyce et al. KDD'25 — examples/lgbm_config.json)
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

# Ten seeds enable Wilcoxon two-sided minimum p ~ 0.00195, sufficient power
# for Benjamini-Hochberg FDR significance at q < 0.05.
SEEDS = [42, 123, 456, 789, 1011, 2026, 3141, 4242, 5555, 6789]
PROTOTYPE_SEEDS = [42, 123]

N_FEATURES = 2568

GROUP_CODES = ["GFI", "BH", "BEH", "STR", "HDR", "SEC", "IMP",
               "EXP", "DD", "RH", "AUTH", "WARN"]

PE_FILE_TYPES = ["win32", "win64", "dot_net"]

FPR_LEVELS = [0.001, 0.01]
PRIMARY_METRIC = "tpr_at_fpr_001_challenge"

VAL_SPLIT = 0.1
SPLIT_SEED = 0  # FIXED across all seeds

ALPHA = 0.05
FDR_METHOD = "fdr_bh"

# Prototype sub-sampling per type (smoke test on Windows local)
PROTOTYPE_PER_TYPE = 5000
