"""peto.py — Per-Expert Threshold Optimization for P01 v2.

Replaces v1's global FPR=1% threshold pick with per-expert thresholds that
maximize aggregate TPR subject to ensemble FPR budget:

    maximize  sum_t  TP_t(theta_t)
    subject to  sum_t FP_t(theta_t) / n_total_benign  <=  fpr_budget
                theta_t in feasible grid based on per-type benign quantiles

Solved via 3D grid search with the inner axis collapsed via the constraint
(O(n_grid^2 * log n_grid)). For n_grid=200, this completes in seconds.

Reference: P18 sweet-zone insight — different classifier families have
different optimal thresholds; here, different per-type experts likewise have
type-specific sweet zones.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _build_threshold_grid(scores_benign: np.ndarray, n_grid: int = 200,
                           fpr_lo: float = 0.0005, fpr_hi: float = 0.05) -> np.ndarray:
    """Geometric grid of thresholds covering FPR ∈ [fpr_lo, fpr_hi] on benign scores."""
    if scores_benign.size == 0:
        raise ValueError("benign scores empty for grid")
    fpr_grid = np.geomspace(fpr_lo, fpr_hi, n_grid)
    return np.array([float(np.quantile(scores_benign, 1.0 - f, method="higher"))
                     for f in fpr_grid])


def _count_above_threshold(sorted_scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """For each threshold t, count scores >= t (vectorized via searchsorted)."""
    return sorted_scores.size - np.searchsorted(sorted_scores, thresholds, side="left")


def compute_peto_thresholds(
    scores_benign_per_type: Dict[str, np.ndarray],
    scores_malware_per_type: Dict[str, np.ndarray],
    *,
    fpr_budget: float = 0.01,
    n_grid: int = 200,
    fpr_lo: float = 0.0005,
    fpr_hi: float = 0.05,
) -> Dict[str, float]:
    """Find per-expert thresholds maximizing aggregate TPR under ensemble FPR budget.

    Args:
        scores_benign_per_type: {file_type: 1-D scores on test benign filtered to type}.
        scores_malware_per_type: {file_type: 1-D scores on challenge/test mal filtered}.
        fpr_budget: ensemble false-positive rate budget (e.g. 0.01 = 1%).
        n_grid: per-type threshold grid resolution (default 200).
        fpr_lo, fpr_hi: per-type FPR band defining the threshold grid.

    Returns:
        thresholds: {file_type: threshold_value}.
    """
    types = list(scores_benign_per_type.keys())
    if len(types) != 3:
        raise NotImplementedError(
            f"PETO grid search currently assumes 3 file-type experts, got {len(types)}"
        )
    if set(scores_malware_per_type.keys()) != set(types):
        raise ValueError("scores_benign_per_type and scores_malware_per_type keys mismatch")

    grids: Dict[str, np.ndarray] = {}
    fp_grid: Dict[str, np.ndarray] = {}
    tp_grid: Dict[str, np.ndarray] = {}
    for t in types:
        b = np.asarray(scores_benign_per_type[t], dtype=np.float64)
        m = np.asarray(scores_malware_per_type[t], dtype=np.float64)
        if b.size == 0 or m.size == 0:
            raise ValueError(f"empty scores for type {t!r}")
        grids[t] = _build_threshold_grid(b, n_grid=n_grid, fpr_lo=fpr_lo, fpr_hi=fpr_hi)
        b_sorted = np.sort(b)
        m_sorted = np.sort(m)
        fp_grid[t] = _count_above_threshold(b_sorted, grids[t])
        tp_grid[t] = _count_above_threshold(m_sorted, grids[t])

    n_benign_total = sum(scores_benign_per_type[t].size for t in types)
    fp_budget_count = int(fpr_budget * n_benign_total)

    t_a, t_b, t_c = types
    best_tpr = -1
    best_idx = (0, 0, 0)
    fp_c_arr = fp_grid[t_c]
    tp_c_arr = tp_grid[t_c]

    for i_a in range(n_grid):
        fp_a = int(fp_grid[t_a][i_a])
        if fp_a > fp_budget_count:
            continue
        tp_a = int(tp_grid[t_a][i_a])
        for i_b in range(n_grid):
            fp_b = int(fp_grid[t_b][i_b])
            remaining = fp_budget_count - fp_a - fp_b
            if remaining < 0:
                continue
            valid = fp_c_arr <= remaining
            if not valid.any():
                continue
            tp_b = int(tp_grid[t_b][i_b])
            tp_c_masked = np.where(valid, tp_c_arr, -1)
            i_c = int(np.argmax(tp_c_masked))
            tpr_total = tp_a + tp_b + int(tp_c_arr[i_c])
            if tpr_total > best_tpr:
                best_tpr = tpr_total
                best_idx = (i_a, i_b, i_c)

    return {t: float(grids[t][best_idx[i]]) for i, t in enumerate(types)}


def evaluate_peto(
    thresholds: Dict[str, float],
    scores_benign_per_type: Dict[str, np.ndarray],
    scores_malware_per_type: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Diagnostic — return per-type TPR, FPR, and ensemble values for a threshold dict."""
    types = list(thresholds.keys())
    out: Dict[str, float] = {}
    fp_total = 0
    tp_total = 0
    n_benign_total = 0
    n_mal_total = 0
    for t in types:
        theta = thresholds[t]
        b = np.asarray(scores_benign_per_type[t])
        m = np.asarray(scores_malware_per_type[t])
        fp_t = int((b >= theta).sum())
        tp_t = int((m >= theta).sum())
        out[f"tpr_{t}"] = float(tp_t) / max(1, m.size)
        out[f"fpr_{t}"] = float(fp_t) / max(1, b.size)
        out[f"threshold_{t}"] = float(theta)
        fp_total += fp_t
        tp_total += tp_t
        n_benign_total += b.size
        n_mal_total += m.size
    out["ensemble_fpr"] = float(fp_total) / max(1, n_benign_total)
    out["ensemble_tpr"] = float(tp_total) / max(1, n_mal_total)
    return out
