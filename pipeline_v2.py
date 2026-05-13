"""pipeline_v2.py — main runner for P01 v2 ablation experiments.

Integrates: FTE (file-type experts) + MG-EP-PT (multi-group EP per-type) +
SAR (state-aware reweighting) + LS (size-aware reweighting from v1) + PETO
(per-expert threshold optimization) + Optuna HPO per type.

Config registry maps 14 ablation cells to boolean flags. Single
`run_config(...)` handles all cells via composition.

Output per (config_id, seed):
    {output_dir}/{config_id}/seed_{S}/
        metrics.json                    # all FPR levels, per-type + ensemble
        predictions_test_benign_{ft}.npz   per file type
        predictions_challenge_{ft}.npz     per file type
        thresholds.json                 # PETO thresholds (or global)
        log.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

import config_v2 as cfg
import multi_group_ep_per_type as mgep
import state_aware_reweighting as sar
import peto as peto_mod
from ember2024_parquet_loader import load_detection, load_challenge_mixed
from ember_v3_schema import SIZE_FEATURE_IDX
from preflight_per_type_lofo.feature_groups import (
    indices_for_group as fg_indices_for_group,
)


logger = logging.getLogger("p01_v2")


# ── Config registry — 14 ablation cells ──────────────────────────────────────

CONFIG_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Baselines (unified model, no FTE)
    "baseline_A": dict(use_fte=False, baseline_params="A"),
    "baseline_B": dict(use_fte=False, baseline_params="B"),

    # FTE alone
    "fte": dict(use_fte=True, baseline_params="A"),

    # FTE + MG-EP-PT variants
    "fte_mgep_top3": dict(use_fte=True, use_mgep=True, mgep_top_k=3,
                          mgep_ranking="per_type", baseline_params="A"),
    "fte_mgep_top5": dict(use_fte=True, use_mgep=True, mgep_top_k=5,
                          mgep_ranking="per_type", baseline_params="A"),

    # FTE + SAR
    "fte_sar": dict(use_fte=True, use_sar=True, baseline_params="A"),

    # FTE + LS (re-implementing paper-Feb-2026 LS for fair ablation)
    "fte_ls": dict(use_fte=True, use_ls=True, ls_alpha=11, ls_percentile=90,
                   baseline_params="A"),

    # Phase 4b winner: FTE + LS + HPO (Optuna 50 trials per file type)
    "fte_ls_hpo": dict(use_fte=True, use_ls=True, ls_alpha=11, ls_percentile=90,
                       use_hpo=True, baseline_params="A"),

    # FTE + LS with fixed Optuna-100-trial hyperparameters (one-time offline tune).
    # Avoids in-loop HPO (~9h/seed) -> approximately 10 min/seed x 10 seeds.
    "fte_ls_optuna100": dict(use_fte=True, use_ls=True, ls_alpha=11, ls_percentile=90,
                              use_hpo=False, baseline_params="OPTUNA100"),

    # FTE + 2-component combos
    "fte_mgep_sar": dict(use_fte=True, use_mgep=True, use_sar=True,
                         baseline_params="A"),

    # FTE + 3-component (no PETO no HPO)
    "fte_mgep_sar_ls": dict(use_fte=True, use_mgep=True, use_sar=True,
                             use_ls=True, ls_alpha=11, ls_percentile=90,
                             baseline_params="A"),

    # FTE + 4-component (PETO, no HPO)
    "fte_mgep_sar_ls_peto_noHPO": dict(
        use_fte=True, use_mgep=True, use_sar=True, use_ls=True,
        ls_alpha=11, ls_percentile=90, use_peto=True, baseline_params="A",
    ),

    # PROPOSED v2 — all 4 components + HPO/type
    "proposed_v2": dict(
        use_fte=True, use_mgep=True, use_sar=True, use_ls=True,
        ls_alpha=11, ls_percentile=90, use_peto=True, use_hpo=True,
        baseline_params="A",
    ),

    # Legacy v1 — EP-on-BEH only (back-compat with paper-Feb-2026 EP)
    "fte_beh_only": dict(
        use_fte=True, use_mgep=True, mgep_ranking="BEH_only",
        baseline_params="A",
    ),

    # Drop-bottom feature ablations
    "drop_bottom_per_type_3": dict(
        use_fte=True, use_mgep=True, use_sar=True, use_ls=True,
        ls_alpha=11, ls_percentile=90, use_peto=True, use_hpo=True,
        drop_bottom="per_type_3", baseline_params="A",
    ),
    "drop_bottom_dotnet_8": dict(
        use_fte=True, use_mgep=True, use_sar=True, use_ls=True,
        ls_alpha=11, ls_percentile=90, use_peto=True, use_hpo=True,
        drop_bottom="dotnet_8", baseline_params="A",
    ),

    # R-clipping sensitivity
    "R_5": dict(
        use_fte=True, use_mgep=True, use_sar=True, use_ls=True,
        ls_alpha=11, ls_percentile=90, use_peto=True, use_hpo=True,
        weight_clip_R=5.0, baseline_params="A",
    ),
    "R_20": dict(
        use_fte=True, use_mgep=True, use_sar=True, use_ls=True,
        ls_alpha=11, ls_percentile=90, use_peto=True, use_hpo=True,
        weight_clip_R=20.0, baseline_params="A",
    ),
}


# ── Base param selector (Baseline-A / B / Optuna-100-trial) ─────────────────

def _select_base_params(baseline_params: Optional[str]) -> Dict[str, Any]:
    """Select base LightGBM params dict based on string identifier."""
    if baseline_params == "A":
        return cfg.BASELINE_A
    if baseline_params == "OPTUNA100":
        return cfg.BASELINE_OPTUNA_100T
    # Default: Baseline-B (Joyce et al. KDD'25 github)
    return cfg.BASELINE_B


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data_per_type(parquet_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load EMBER2024 PE data per file type. Returns dict keyed by file type."""
    data: Dict[str, Dict[str, Any]] = {}
    for ft in cfg.PE_FILE_TYPES:
        t0 = time.perf_counter()
        X_tr, y_tr, meta_tr = load_detection(str(parquet_dir), "train", [ft])
        X_te_b, y_te_b, X_ch, y_ch, meta_tb, meta_ch = load_challenge_mixed(
            str(parquet_dir), [ft]
        )
        sha_tr = np.array(meta_tr["sha256"].to_list())
        size_tr = X_tr[:, SIZE_FEATURE_IDX].astype(np.float64)
        data[ft] = dict(
            X_train=X_tr, y_train=y_tr, sha256_train=sha_tr,
            file_size_train=size_tr,
            X_test_benign=X_te_b,
            X_challenge=X_ch,
        )
        logger.info(
            "[LOAD %s] train=%d (mal=%d ben=%d) test_ben=%d challenge=%d (%.1fs)",
            ft, len(y_tr), int(y_tr.sum()), int((y_tr == 0).sum()),
            len(y_te_b), len(X_ch), time.perf_counter() - t0,
        )
    return data


def load_states_for_data(
    data: Dict[str, Dict[str, Any]], relabel_frame_path: Path,
) -> None:
    """Mutate `data` in-place adding 'states' key per file type."""
    all_sha = np.concatenate([data[ft]["sha256_train"] for ft in cfg.PE_FILE_TYPES])
    t0 = time.perf_counter()
    all_states = sar.load_state_labels_for_sha256(
        all_sha, parquet_path=str(relabel_frame_path),
    )
    offset = 0
    for ft in cfg.PE_FILE_TYPES:
        n = len(data[ft]["sha256_train"])
        data[ft]["states"] = all_states[offset:offset + n]
        offset += n
    summary = sar.state_distribution_summary(all_states)
    logger.info("[LOAD states] %.1fs all-types: %s", time.perf_counter() - t0, summary)


# ── Weighting composition ────────────────────────────────────────────────────

def compute_ls_weights(
    file_size: np.ndarray, y: np.ndarray,
    alpha: float = 11.0, percentile: float = 90.0,
) -> np.ndarray:
    """LS reweighting (paper-Feb-2026 size-aware). Malicious + size > Pp → 1+alpha."""
    mal_sizes = file_size[y == 1]
    if len(mal_sizes) == 0:
        return np.ones(len(y), dtype=np.float64)
    threshold = float(np.percentile(mal_sizes, percentile))
    weights = np.ones(len(y), dtype=np.float64)
    boost_mask = (y == 1) & (file_size > threshold)
    weights[boost_mask] = 1.0 + alpha
    return weights


def compute_combined_weights(
    X_train: np.ndarray, y_train: np.ndarray,
    X_challenge: np.ndarray,
    file_size: Optional[np.ndarray],
    states: Optional[np.ndarray],
    file_type: str,
    *,
    use_mgep: bool = False,
    mgep_top_k: int = 3,
    mgep_ranking: str = "per_type",
    use_sar: bool = False,
    use_ls: bool = False,
    ls_alpha: float = 11.0,
    ls_percentile: float = 90.0,
    weight_clip_R: float = 10.0,
) -> np.ndarray:
    """Compose MG-EP-PT × SAR × LS weights with clipping. Returns w of shape (n,)."""
    n = len(y_train)
    w = np.ones(n, dtype=np.float64)

    if use_mgep:
        if mgep_ranking == "per_type":
            w_mgep = mgep.compute_mg_ep_weights(
                X_train, y_train, X_challenge, file_type,
                weight_mode="delta_tpr_weighted",
            )
        elif mgep_ranking == "BEH_only":
            # Legacy v1 EP using only BEH group
            w_mgep = _compute_beh_only_ep_weights(X_train, y_train, X_challenge)
        elif mgep_ranking == "global":
            # Use global P0Y top-3: STR, HDR, SEC (uniform alpha)
            w_mgep = _compute_global_top3_ep_weights(X_train, y_train, X_challenge)
        else:
            raise ValueError(f"unknown mgep_ranking={mgep_ranking!r}")
        w = w * w_mgep

    if use_sar:
        if states is None:
            raise ValueError("use_sar=True but states not provided")
        w_sar = sar.compute_sar_weights(
            states,
            k_late_confirmed=cfg.SAR_K_LATE_CONFIRMED,
            k_reverted=cfg.SAR_K_REVERTED,
        )
        w = w * w_sar

    if use_ls:
        if file_size is None:
            raise ValueError("use_ls=True but file_size not provided")
        w_ls = compute_ls_weights(file_size, y_train, alpha=ls_alpha,
                                   percentile=ls_percentile)
        w = w * w_ls

    # Clip combined (only if any reweighting was applied)
    if use_mgep or use_sar or use_ls:
        w = mgep.clip_weights(w, R=weight_clip_R)

    return w


def _compute_beh_only_ep_weights(
    X_train: np.ndarray, y_train: np.ndarray, X_challenge: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """Legacy v1 EP: cosine to challenge BEH centroid only."""
    centroid = mgep.compute_challenge_centroid(X_challenge, "BEH")
    idx = fg_indices_for_group("BEH")
    sims = mgep._cosine_similarity_clipped(X_train[:, idx], centroid)
    weights = np.ones(len(y_train), dtype=np.float64)
    is_mal = (y_train.astype(bool))
    weights[is_mal] += beta * sims[is_mal]
    return weights


def _compute_global_top3_ep_weights(
    X_train: np.ndarray, y_train: np.ndarray, X_challenge: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """Multi-group EP using P0Y GLOBAL top-3 (STR, HDR, SEC) with uniform alpha."""
    groups = ["STR", "HDR", "SEC"]
    n = len(y_train)
    accum = np.zeros(n, dtype=np.float64)
    for g in groups:
        idx = fg_indices_for_group(g)
        centroid = mgep.compute_challenge_centroid(X_challenge, g)
        sims = mgep._cosine_similarity_clipped(X_train[:, idx], centroid)
        accum += sims
    weights = np.ones(n, dtype=np.float64)
    is_mal = (y_train.astype(bool))
    weights[is_mal] += beta * accum[is_mal]
    return weights


# ── Feature pruning (drop_bottom) ────────────────────────────────────────────

def apply_drop_bottom(
    X_train: np.ndarray, X_test_benign: np.ndarray, X_challenge: np.ndarray,
    file_type: str, mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Drop specific groups from features. Returns reduced X arrays + remaining
    categorical feature indices (re-mapped after drop)."""
    if mode == "per_type_3":
        groups_to_drop = cfg.DROP_BOTTOM_3_PER_TYPE[file_type]
    elif mode == "dotnet_8":
        if file_type != "dot_net":
            return X_train, X_test_benign, X_challenge, list(cfg.BASELINE_B.get("categorical_feature", []))
        groups_to_drop = cfg.DROP_BOTTOM_8_DOTNET
    else:
        raise ValueError(f"unknown drop_bottom mode={mode!r}")

    keep_idx_set = set(range(2568))
    for g in groups_to_drop:
        for fi in fg_indices_for_group(g):
            keep_idx_set.discard(int(fi))
    keep_idx = sorted(keep_idx_set)
    remap = {old: new for new, old in enumerate(keep_idx)}
    cat_remapped = [remap[c] for c in [2, 3, 4, 5, 6, 701, 702] if c in remap]

    return (X_train[:, keep_idx],
            X_test_benign[:, keep_idx],
            X_challenge[:, keep_idx],
            cat_remapped)


# ── Train + eval helpers ─────────────────────────────────────────────────────

def train_lightgbm(
    X_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray,
    params: Dict[str, Any], num_threads: int = 60,
    categorical_features: Optional[List[int]] = None,
    seed: int = 42,
) -> Any:
    """Train LightGBM with sample weights + 10% stratified val for early stopping."""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, sample_weight,
        test_size=cfg.VAL_SPLIT, random_state=cfg.SPLIT_SEED, stratify=y_train,
    )

    cat = categorical_features if categorical_features is not None else [2, 3, 4, 5, 6, 701, 702]
    train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                             categorical_feature=cat, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, weight=w_val,
                           categorical_feature=cat, reference=train_set)

    p = dict(params)
    p["seed"] = seed
    p["num_threads"] = num_threads
    p["verbose"] = -1
    p["metric"] = "auc"

    booster = lgb.train(
        p, train_set,
        num_boost_round=p.get("n_estimators", 500),
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return booster


def predict_proba(booster, X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        return np.array([], dtype=np.float64)
    return booster.predict(X, num_iteration=booster.best_iteration).astype(np.float64)


def threshold_for_fpr(scores_benign: np.ndarray, fpr: float) -> float:
    """Smallest threshold such that FPR ≤ target."""
    return float(np.quantile(scores_benign, 1.0 - fpr, method="higher"))


# ── HPO (Optuna per type) ────────────────────────────────────────────────────

def run_optuna_hpo(
    X_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray,
    X_challenge: np.ndarray,
    base_params: Dict[str, Any], num_threads: int, seed: int,
    n_trials: int = 50,
    categorical_features: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Optuna TPE Bayesian HPO. Objective = val challenge-proxy TPR@1%FPR.

    For HPO we use a held-out fold from training (10% val) — challenge set is
    NOT touched here to avoid leakage. The val set serves as proxy.
    """
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import train_test_split

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, sample_weight,
        test_size=cfg.VAL_SPLIT, random_state=cfg.SPLIT_SEED, stratify=y_train,
    )
    cat = categorical_features if categorical_features is not None else [2, 3, 4, 5, 6, 701, 702]

    val_benign_idx = (y_val == 0)
    val_mal_idx = (y_val == 1)

    def objective(trial: "optuna.Trial") -> float:
        p = dict(base_params)
        p["n_estimators"] = trial.suggest_int("n_estimators", 500, 2000, step=100)
        p["num_leaves"] = trial.suggest_int("num_leaves", 32, 128)
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        p["feature_fraction"] = trial.suggest_float("feature_fraction", 0.6, 1.0)
        p["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.6, 1.0)
        p["bagging_freq"] = 1
        p["seed"] = seed
        p["num_threads"] = num_threads
        p["verbose"] = -1

        train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                                  categorical_feature=cat, free_raw_data=False)
        val_set = lgb.Dataset(X_val, label=y_val, weight=w_val,
                                categorical_feature=cat, reference=train_set)
        booster = lgb.train(
            p, train_set, num_boost_round=p["n_estimators"],
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        scores_val = booster.predict(X_val, num_iteration=booster.best_iteration)
        scores_b = scores_val[val_benign_idx]
        scores_m = scores_val[val_mal_idx]
        if len(scores_b) == 0 or len(scores_m) == 0:
            return 0.0
        thresh = threshold_for_fpr(scores_b, 0.01)
        tpr = float(np.mean(scores_m >= thresh))
        return tpr

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(base_params)
    best_params.update(study.best_params)
    best_params["bagging_freq"] = 1
    return best_params


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    scores_benign_per_type: Dict[str, np.ndarray],
    scores_malware_per_type: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score, average_precision_score

    out: Dict[str, Any] = {"per_type": {}, "ensemble": {}}
    fp_total = 0
    tp_total = 0
    n_b_total = 0
    n_m_total = 0
    all_b_scores: List[float] = []
    all_m_scores: List[float] = []

    for ft in scores_benign_per_type:
        sb = scores_benign_per_type[ft]
        sm = scores_malware_per_type[ft]
        theta = thresholds[ft]
        fp = int((sb >= theta).sum())
        tp = int((sm >= theta).sum())
        # Per-type metrics
        per_type_metrics = {
            "n_benign": int(sb.size),
            "n_challenge_mal": int(sm.size),
            "threshold_at_fpr_0.01": float(theta),
            "tpr_at_fpr_0.01_challenge": float(tp) / max(1, sm.size),
            "fpr_at_threshold": float(fp) / max(1, sb.size),
        }
        # Also compute TPR@FPR=0.001 per type via threshold at type-level
        if sb.size > 0:
            theta_p1 = threshold_for_fpr(sb, 0.001)
            per_type_metrics["tpr_at_fpr_0.001_challenge"] = float((sm >= theta_p1).mean())
        # ROC + PR AUC require both labels
        y_combined = np.concatenate([np.zeros_like(sb), np.ones_like(sm)])
        s_combined = np.concatenate([sb, sm])
        if len(np.unique(y_combined)) == 2:
            per_type_metrics["roc_auc_challenge_mixed"] = float(
                roc_auc_score(y_combined, s_combined)
            )
            per_type_metrics["pr_auc_challenge_mixed"] = float(
                average_precision_score(y_combined, s_combined)
            )
        out["per_type"][ft] = per_type_metrics

        fp_total += fp
        tp_total += tp
        n_b_total += sb.size
        n_m_total += sm.size
        all_b_scores.extend(sb.tolist())
        all_m_scores.extend(sm.tolist())

    # Ensemble metrics (concatenated all-PE)
    sb_all = np.array(all_b_scores)
    sm_all = np.array(all_m_scores)
    out["ensemble"] = {
        "n_benign": int(n_b_total),
        "n_challenge_mal": int(n_m_total),
        "ensemble_fpr": float(fp_total) / max(1, n_b_total),
        "ensemble_tpr_at_fpr_0.01": float(tp_total) / max(1, n_m_total),
    }
    # Global TPR@FPR=0.01 with single concatenated threshold
    if sb_all.size > 0:
        theta_global_001 = threshold_for_fpr(sb_all, 0.01)
        out["ensemble"]["global_threshold_at_fpr_0.01"] = float(theta_global_001)
        out["ensemble"]["global_tpr_at_fpr_0.01"] = float((sm_all >= theta_global_001).mean())
        theta_global_p1 = threshold_for_fpr(sb_all, 0.001)
        out["ensemble"]["global_tpr_at_fpr_0.001"] = float((sm_all >= theta_global_p1).mean())
        from sklearn.metrics import roc_auc_score, average_precision_score
        y_ens = np.concatenate([np.zeros_like(sb_all), np.ones_like(sm_all)])
        s_ens = np.concatenate([sb_all, sm_all])
        out["ensemble"]["roc_auc_ensemble"] = float(roc_auc_score(y_ens, s_ens))
        out["ensemble"]["pr_auc_ensemble"] = float(average_precision_score(y_ens, s_ens))
    return out


# ── Main run_config ──────────────────────────────────────────────────────────

def run_config(
    config_id: str, seed: int, parquet_dir: Path, relabel_frame: Path,
    output_dir: Path, num_threads: int = 60, hpo_n_trials: int = 50,
    no_hpo: bool = False, with_hpo: bool = False, save_model: bool = False,
) -> Dict[str, Any]:
    if config_id not in CONFIG_REGISTRY:
        raise ValueError(f"unknown config_id {config_id!r}; available: {list(CONFIG_REGISTRY)}")
    kw = dict(CONFIG_REGISTRY[config_id])  # copy to avoid mutating registry
    if no_hpo:
        kw["use_hpo"] = False
    if with_hpo:
        kw["use_hpo"] = True
    out_dir = output_dir / config_id / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "log.txt", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    started = datetime.now(timezone.utc).isoformat()
    logger.info("=== run_config %s seed=%d started %s ===", config_id, seed, started)
    logger.info("kwargs: %s", kw)
    t_total = time.perf_counter()

    # 1. Load data per type
    data = load_data_per_type(parquet_dir)

    # 2. Load states if needed
    if kw.get("use_sar"):
        load_states_for_data(data, relabel_frame)

    # 3. Train + predict
    scores_b: Dict[str, np.ndarray] = {}
    scores_m: Dict[str, np.ndarray] = {}

    if not kw.get("use_fte", True):
        # Unified model — concat all 3 types
        X_train_all = np.concatenate([data[ft]["X_train"] for ft in cfg.PE_FILE_TYPES])
        y_train_all = np.concatenate([data[ft]["y_train"] for ft in cfg.PE_FILE_TYPES])
        weights = np.ones(len(y_train_all), dtype=np.float64)
        params = _select_base_params(kw.get("baseline_params"))
        t0 = time.perf_counter()
        booster = train_lightgbm(X_train_all, y_train_all, weights, params,
                                  num_threads=num_threads, seed=seed)
        logger.info("[unified] trained in %.1fs", time.perf_counter() - t0)
        if save_model:
            booster.save_model(str(out_dir / "model_unified.txt"))
            logger.info("[unified] model saved to %s", out_dir / "model_unified.txt")
        for ft in cfg.PE_FILE_TYPES:
            scores_b[ft] = predict_proba(booster, data[ft]["X_test_benign"])
            scores_m[ft] = predict_proba(booster, data[ft]["X_challenge"])
    else:
        # Per-type expert
        for ft in cfg.PE_FILE_TYPES:
            d = data[ft]
            X_tr_use, X_te_b_use, X_ch_use = d["X_train"], d["X_test_benign"], d["X_challenge"]
            cat_use: Optional[List[int]] = None
            if kw.get("drop_bottom"):
                X_tr_use, X_te_b_use, X_ch_use, cat_use = apply_drop_bottom(
                    X_tr_use, X_te_b_use, X_ch_use, ft, kw["drop_bottom"],
                )
                logger.info("[drop_bottom %s] %s features kept", ft, X_tr_use.shape[1])

            weights = compute_combined_weights(
                X_tr_use, d["y_train"], X_ch_use,
                file_size=d["file_size_train"] if kw.get("use_ls") else None,
                states=d.get("states") if kw.get("use_sar") else None,
                file_type=ft,
                use_mgep=kw.get("use_mgep", False),
                mgep_top_k=kw.get("mgep_top_k", 3),
                mgep_ranking=kw.get("mgep_ranking", "per_type"),
                use_sar=kw.get("use_sar", False),
                use_ls=kw.get("use_ls", False),
                ls_alpha=kw.get("ls_alpha", 11.0),
                ls_percentile=kw.get("ls_percentile", 90.0),
                weight_clip_R=kw.get("weight_clip_R", 10.0),
            )
            base_params = _select_base_params(kw.get("baseline_params"))
            if kw.get("use_hpo"):
                t0 = time.perf_counter()
                params = run_optuna_hpo(
                    X_tr_use, d["y_train"], weights, X_ch_use,
                    base_params=base_params, num_threads=num_threads, seed=seed,
                    n_trials=hpo_n_trials, categorical_features=cat_use,
                )
                logger.info("[%s HPO] best params in %.1fs", ft, time.perf_counter() - t0)
            else:
                params = base_params

            t0 = time.perf_counter()
            booster = train_lightgbm(X_tr_use, d["y_train"], weights, params,
                                      num_threads=num_threads,
                                      categorical_features=cat_use, seed=seed)
            logger.info("[%s expert] trained in %.1fs (n=%d)",
                        ft, time.perf_counter() - t0, len(d["y_train"]))
            if save_model:
                booster.save_model(str(out_dir / f"model_{ft}.txt"))
                logger.info("[%s] model saved", ft)
            scores_b[ft] = predict_proba(booster, X_te_b_use)
            scores_m[ft] = predict_proba(booster, X_ch_use)

    # 4. PETO or global threshold
    if kw.get("use_peto"):
        t0 = time.perf_counter()
        thresholds = peto_mod.compute_peto_thresholds(
            scores_b, scores_m, fpr_budget=cfg.PETO_FPR_BUDGET,
        )
        logger.info("[PETO] thresholds in %.1fs: %s", time.perf_counter() - t0, thresholds)
    else:
        # Global threshold across all types
        all_b = np.concatenate(list(scores_b.values()))
        thresh_global = threshold_for_fpr(all_b, 0.01)
        thresholds = {ft: thresh_global for ft in cfg.PE_FILE_TYPES}

    # 5. Compute + save metrics
    metrics = compute_metrics(scores_b, scores_m, thresholds)
    metrics["config_id"] = config_id
    metrics["seed"] = seed
    metrics["started_utc"] = started
    metrics["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metrics["wallclock_s"] = time.perf_counter() - t_total
    metrics["kwargs"] = kw
    metrics["thresholds"] = thresholds

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8",
    )
    for ft in cfg.PE_FILE_TYPES:
        np.savez_compressed(out_dir / f"predictions_test_benign_{ft}.npz",
                             scores=scores_b[ft])
        np.savez_compressed(out_dir / f"predictions_challenge_{ft}.npz",
                             scores=scores_m[ft])
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2),
                                              encoding="utf-8")
    logger.info("=== DONE %s seed=%d in %.1fs — ensemble TPR@1%%FPR=%.4f ===",
                config_id, seed, metrics["wallclock_s"],
                metrics["ensemble"].get("ensemble_tpr_at_fpr_0.01", -1))
    logger.removeHandler(fh)
    fh.close()
    return metrics


def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_id", required=True,
                    help=f"One of: {list(CONFIG_REGISTRY)}")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--relabel_frame", default="",
                    help="P11+ relabel_frame.parquet path (required if SAR used)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_threads", type=int, default=60)
    ap.add_argument("--hpo_n_trials", type=int, default=50)
    ap.add_argument("--no_hpo", action="store_true",
                    help="Override config's use_hpo to False (Phase 4a no-HPO ablation)")
    ap.add_argument("--with_hpo", action="store_true",
                    help="Override config's use_hpo to True (Phase 4b HPO on winner config)")
    ap.add_argument("--save_model", action="store_true",
                    help="Save trained LightGBM booster (.txt format) for feature_importance + perturbation analysis")
    return ap.parse_args()


def main() -> int:
    args = parse_cli()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parquet_dir = Path(args.parquet_dir)
    relabel_frame = Path(args.relabel_frame) if args.relabel_frame else Path("")
    output_dir = Path(args.output_dir)
    run_config(
        config_id=args.config_id, seed=args.seed, parquet_dir=parquet_dir,
        relabel_frame=relabel_frame, output_dir=output_dir,
        num_threads=args.num_threads, hpo_n_trials=args.hpo_n_trials,
        no_hpo=args.no_hpo, with_hpo=args.with_hpo, save_model=args.save_model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
