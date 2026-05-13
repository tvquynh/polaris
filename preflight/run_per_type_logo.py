#!/usr/bin/env python3
"""
run_per_type_logo.py — Per-type group LOGO for P01 v2 preflight.

Per file type (win32/win64/dot_net):
  1. Filter train/test/challenge to that type.
  2. Stratified 90/10 train/val split (split_seed fixed).
  3. Train baseline (full features) → save full_metrics_<type>.json
  4. For each of 12 groups:
       train with kept_indices = indices_excluding_group(g)
       eval on type-filtered test_benign + test_malware + challenge_mal
       save group_metrics_<type>.json (incremental, resume-safe)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

_THIS = Path(__file__).resolve().parent
_P01 = _THIS.parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
if str(_P01) not in sys.path:
    sys.path.insert(0, str(_P01))

from feature_groups import indices_excluding_group, categorical_features_after_drop
from config_preflight import (
    BASELINE_B, GROUP_CODES, PE_FILE_TYPES, FPR_LEVELS,
    VAL_SPLIT, SPLIT_SEED, PROTOTYPE_PER_TYPE,
)
from ember_v3_schema import assert_schema  # from p01 parent
from ember2024_parquet_loader import load_detection, load_challenge_mixed  # from p01 parent


# ── Metrics (vendored minimal) ──

def threshold_for_fpr(scores_benign: np.ndarray, fpr_target: float) -> float:
    if scores_benign.size == 0:
        raise ValueError("scores_benign empty")
    q = 1.0 - fpr_target
    try:
        return float(np.quantile(scores_benign, q, method="higher"))
    except TypeError:
        return float(np.quantile(scores_benign, q, interpolation="higher"))


def tpr_at_threshold(scores: np.ndarray, threshold: float) -> float:
    if scores.size == 0:
        return float("nan")
    return float((scores >= threshold).mean())


def compute_metrics(val_benign: np.ndarray, test_benign: np.ndarray,
                     test_malware: np.ndarray, challenge: np.ndarray,
                     fpr_levels=(0.001, 0.01)) -> dict:
    t1 = threshold_for_fpr(val_benign, fpr_levels[0])
    t2 = threshold_for_fpr(val_benign, fpr_levels[1])
    out = {
        "tpr_at_fpr_0001_regular": tpr_at_threshold(test_malware, t1),
        "tpr_at_fpr_0001_challenge": tpr_at_threshold(challenge, t1),
        "tpr_at_fpr_001_regular": tpr_at_threshold(test_malware, t2),
        "tpr_at_fpr_001_challenge": tpr_at_threshold(challenge, t2),
        "actual_fpr_at_t_for_0001": tpr_at_threshold(test_benign, t1),
        "actual_fpr_at_t_for_001": tpr_at_threshold(test_benign, t2),
    }
    if test_benign.size and test_malware.size:
        y = np.concatenate([np.zeros_like(test_benign), np.ones_like(test_malware)])
        s = np.concatenate([test_benign, test_malware])
        out["roc_auc_regular"] = float(roc_auc_score(y, s))
        out["pr_auc_regular"] = float(average_precision_score(y, s))
    if test_benign.size and challenge.size:
        y = np.concatenate([np.zeros_like(test_benign), np.ones_like(challenge)])
        s = np.concatenate([test_benign, challenge])
        out["roc_auc_challenge"] = float(roc_auc_score(y, s))
        out["pr_auc_challenge"] = float(average_precision_score(y, s))
    out["mean_score_test_benign"] = float(test_benign.mean()) if test_benign.size else float("nan")
    out["mean_score_test_malware"] = float(test_malware.mean()) if test_malware.size else float("nan")
    out["mean_score_challenge"] = float(challenge.mean()) if challenge.size else float("nan")
    return out


# ── LightGBM trainer ──

def train_lightgbm(X_tr: np.ndarray, y_tr: np.ndarray,
                    X_va: np.ndarray, y_va: np.ndarray,
                    kept_indices: Optional[np.ndarray],
                    seed: int, num_threads: int) -> lgb.Booster:
    if kept_indices is None:
        Xtr, Xva = X_tr, X_va
        cat = categorical_features_after_drop(np.arange(X_tr.shape[1]))
    else:
        Xtr, Xva = X_tr[:, kept_indices], X_va[:, kept_indices]
        cat = categorical_features_after_drop(kept_indices)
    p = dict(BASELINE_B)
    p["seed"] = int(seed)
    p["bagging_seed"] = int(seed)
    p["feature_fraction_seed"] = int(seed)
    p["data_random_seed"] = int(seed)
    if num_threads > 0:
        p["num_threads"] = int(num_threads)
    train_set = lgb.Dataset(Xtr, label=y_tr, categorical_feature=cat, free_raw_data=True)
    val_set = lgb.Dataset(Xva, label=y_va, reference=train_set,
                          categorical_feature=cat, free_raw_data=True)
    booster = lgb.train(p, train_set, num_boost_round=p["n_estimators"],
                        valid_sets=[val_set], valid_names=["val"], callbacks=[])
    return booster


def predict(booster: lgb.Booster, X: np.ndarray,
            kept_indices: Optional[np.ndarray] = None) -> np.ndarray:
    Xp = X if kept_indices is None else X[:, kept_indices]
    return booster.predict(Xp).astype(np.float32)


# ── Data loading per type ──

def _ft_array(meta) -> np.ndarray:
    return np.array([v.lower() for v in meta["file_type"].to_list()])


def load_per_type(parquet_dir: str, prototype: bool, logger: logging.Logger) -> Dict[str, dict]:
    logger.info("[LOAD] train")
    X_tr_all, y_tr_all, meta_tr = load_detection(parquet_dir, "train", file_types=PE_FILE_TYPES)
    assert_schema(X_tr_all)
    ft_tr = _ft_array(meta_tr)

    logger.info("[LOAD] test")
    X_te_all, y_te_all, meta_te = load_detection(parquet_dir, "test", file_types=PE_FILE_TYPES)
    assert_schema(X_te_all)
    ft_te = _ft_array(meta_te)

    logger.info("[LOAD] challenge mixed")
    X_tb, y_tb, X_ch_mal, y_ch_mal, meta_tb, meta_ch = load_challenge_mixed(
        parquet_dir, file_types=PE_FILE_TYPES,
    )
    assert_schema(X_tb)
    assert_schema(X_ch_mal)
    ft_tb = _ft_array(meta_tb)
    ft_ch = _ft_array(meta_ch)

    out = {}
    for ft in PE_FILE_TYPES:
        m_tr = ft_tr == ft
        m_te = ft_te == ft
        m_tb = ft_tb == ft
        m_ch = ft_ch == ft
        X_tr = X_tr_all[m_tr]
        y_tr = y_tr_all[m_tr]
        X_te_ben = X_te_all[m_te & (y_te_all == 0)]
        X_te_mal = X_te_all[m_te & (y_te_all == 1)]
        X_ch_ben = X_tb[m_tb]
        X_ch_mal_ft = X_ch_mal[m_ch]
        if prototype and X_tr.shape[0] > PROTOTYPE_PER_TYPE:
            rs = np.random.RandomState(SPLIT_SEED)
            sel = rs.choice(X_tr.shape[0], PROTOTYPE_PER_TYPE, replace=False)
            X_tr = X_tr[sel]
            y_tr = y_tr[sel]
        out[ft] = {
            "X_train": X_tr, "y_train": y_tr,
            "X_test_benign": X_ch_ben,
            "X_test_malware": X_te_mal,
            "X_challenge": X_ch_mal_ft,
        }
        logger.info(
            f"[LOAD {ft}] train={X_tr.shape[0]:,} (mal={int(y_tr.sum()):,}) "
            f"test_ben={X_ch_ben.shape[0]:,} test_mal={X_te_mal.shape[0]:,} "
            f"challenge_mal={X_ch_mal_ft.shape[0]:,}"
        )
    return out


def split_train_val(X: np.ndarray, y: np.ndarray, split_seed: int):
    return train_test_split(X, y, test_size=VAL_SPLIT,
                             random_state=split_seed, stratify=y)


# ── Per-seed driver ──

def run_one_type_one_seed(ft: str, data_ft: dict, seed: int, num_threads: int,
                           out_dir: Path, logger: logging.Logger) -> None:
    Xtr_, Xva_, ytr_, yva_ = split_train_val(data_ft["X_train"], data_ft["y_train"], SPLIT_SEED)
    logger.info(f"[{ft} seed={seed}] split: train={Xtr_.shape[0]:,} val={Xva_.shape[0]:,}")

    full_path = out_dir / f"full_metrics_{ft}.json"
    if not full_path.exists():
        t0 = time.perf_counter()
        booster = train_lightgbm(Xtr_, ytr_, Xva_, yva_, None, seed, num_threads)
        s_val_ben = predict(booster, Xva_[yva_ == 0])
        s_te_ben = predict(booster, data_ft["X_test_benign"])
        s_te_mal = predict(booster, data_ft["X_test_malware"])
        s_ch = predict(booster, data_ft["X_challenge"])
        m = compute_metrics(s_val_ben, s_te_ben, s_te_mal, s_ch, fpr_levels=tuple(FPR_LEVELS))
        train_sec = time.perf_counter() - t0
        full_path.write_text(json.dumps({
            "experiment": "full",
            "file_type": ft,
            "seed": int(seed),
            "n_features": int(Xtr_.shape[1]),
            "train_sec": train_sec,
            "metrics": m,
        }, indent=2))
        logger.info(
            f"[{ft} FULL] {train_sec:.1f}s "
            f"TPR_ch@1%={m['tpr_at_fpr_001_challenge']:.4f}"
        )
        del booster, s_val_ben, s_te_ben, s_te_mal, s_ch
    else:
        logger.info(f"[{ft} FULL] resume — already done")

    grp_path = out_dir / f"group_metrics_{ft}.json"
    grp_out: dict = {}
    if grp_path.exists():
        try:
            grp_out = json.loads(grp_path.read_text())
            logger.info(f"[{ft} resume] {len(grp_out)}/12 groups already done")
        except Exception:
            grp_out = {}

    for i, code in enumerate(GROUP_CODES):
        if code in grp_out:
            continue
        kept = indices_excluding_group(code)
        t0 = time.perf_counter()
        booster = train_lightgbm(Xtr_, ytr_, Xva_, yva_, kept, seed, num_threads)
        s_val_ben = predict(booster, Xva_[yva_ == 0], kept)
        s_te_ben = predict(booster, data_ft["X_test_benign"], kept)
        s_te_mal = predict(booster, data_ft["X_test_malware"], kept)
        s_ch = predict(booster, data_ft["X_challenge"], kept)
        m = compute_metrics(s_val_ben, s_te_ben, s_te_mal, s_ch, fpr_levels=tuple(FPR_LEVELS))
        train_sec = time.perf_counter() - t0
        grp_out[code] = {"train_sec": train_sec, "metrics": m}
        grp_path.write_text(json.dumps(grp_out, indent=2))
        logger.info(
            f"[{ft} GROUP {i+1}/12 {code}] {train_sec:.1f}s "
            f"TPR_ch@1%={m['tpr_at_fpr_001_challenge']:.4f}"
        )
        del booster, s_val_ben, s_te_ben, s_te_mal, s_ch


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-type group LOGO preflight for P01 v2")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prototype", action="store_true")
    ap.add_argument("--num_threads", type=int, default=-1)
    ap.add_argument("--file_types", default=",".join(PE_FILE_TYPES))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"

    logger = logging.getLogger("preflight")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    h_file = logging.FileHandler(log_path, encoding="utf-8")
    h_file.setFormatter(fmt)
    h_console = logging.StreamHandler()
    h_console.setFormatter(fmt)
    logger.addHandler(h_file)
    logger.addHandler(h_console)

    logger.info(f"=== preflight per-type LOGO seed={args.seed} prototype={args.prototype} ===")
    file_types = [ft.strip() for ft in args.file_types.split(",") if ft.strip()]
    for ft in file_types:
        if ft not in PE_FILE_TYPES:
            raise ValueError(f"unknown file_type {ft}")

    data = load_per_type(args.parquet_dir, args.prototype, logger)

    for ft in file_types:
        run_one_type_one_seed(ft, data[ft], args.seed, args.num_threads, out_dir, logger)

    logger.info("=== preflight DONE ===")


if __name__ == "__main__":
    main()
