"""multi_group_ep_per_type.py — MG-EP-PT reweighting for P01 v2.

Computes sample weights for FTE training by combining cosine similarities to
PER-TYPE challenge centroids over top-K feature groups.

Formula (Eq. 1 P01 v2):
    w_mgep_i = 1 + beta * I[y_i=1] * sum_{g in top_K(type_i)}(
                  alpha_g(type_i) * max(0, cos_sim(X_i[g], mu_challenge_{type_i}[g]))
              )

where:
    - top_K(type_i) is locked per file type from preflight LOGO ranking.
    - mu_challenge_{type_i}[g] = mean of challenge malware samples
      (filtered to type_i) over the feature subset belonging to group g.
    - alpha_g normalized so that uniform-mode sum equals K (preserves scale).

For benign samples (y_i=0), weights = 1.0 (no upweighting).

Group indices come from the EMBER2024 v3 schema via feature_groups.py
(located in preflight_per_type_lofo/, which itself imports ember_v3_schema).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

# Reuse feature_groups vendored in preflight folder.
_THIS = Path(__file__).resolve().parent
_PREFLIGHT = _THIS / "preflight_per_type_lofo"
if str(_PREFLIGHT) not in sys.path:
    sys.path.insert(0, str(_PREFLIGHT))
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from feature_groups import indices_for_group  # noqa: E402

from config_v2 import (  # noqa: E402
    TOP_K_PER_TYPE,
    MG_EP_ALPHA_PER_TYPE_TOP3,
    PE_FILE_TYPES,
)


def compute_challenge_centroid(
    X_challenge_type: np.ndarray,
    group: str,
) -> np.ndarray:
    """Mean over challenge samples (already type-filtered) for a given group.

    Args:
        X_challenge_type: (n_challenge_type, 2568) feature matrix for challenge
            malware of one file type.
        group: group code (e.g. "STR", "HDR", "SEC").

    Returns:
        centroid: 1-D array of length |group features|.
    """
    if X_challenge_type.shape[0] == 0:
        raise ValueError(f"empty X_challenge for centroid of group {group}")
    idx = indices_for_group(group)
    return X_challenge_type[:, idx].mean(axis=0).astype(np.float64)


def compute_centroids_for_type(
    X_challenge_type: np.ndarray,
    groups: Sequence[str],
) -> Dict[str, np.ndarray]:
    return {g: compute_challenge_centroid(X_challenge_type, g) for g in groups}


def _cosine_similarity_clipped(
    X_block: np.ndarray, centroid: np.ndarray, eps: float = 1e-12,
) -> np.ndarray:
    """Cosine similarity per row, clipped to [0, 1] (negative → 0).

    Args:
        X_block: (n, d) — training rows on a single feature group.
        centroid: (d,) — challenge centroid on same group.

    Returns:
        sims: (n,) clipped cosine similarities.
    """
    # Cast to float64 to avoid overflow when group contains large-magnitude
    # features (e.g., file_size at feature_0000 ~ 10^8 bytes → square ~ 10^16
    # would overflow float32 norm computation across 256+ features).
    X64 = X_block.astype(np.float64, copy=False)
    centroid64 = centroid.astype(np.float64, copy=False)
    cn = np.linalg.norm(centroid64)
    if cn < eps:
        return np.zeros(X_block.shape[0], dtype=np.float64)
    rn = np.linalg.norm(X64, axis=1)
    safe = rn >= eps
    sims = np.zeros(X_block.shape[0], dtype=np.float64)
    sims[safe] = (X64[safe] @ centroid64) / (rn[safe] * cn)
    return np.clip(sims, 0.0, 1.0)


def compute_mg_ep_weights(
    X_train_type: np.ndarray,
    y_train_type: np.ndarray,
    X_challenge_type: np.ndarray,
    file_type: str,
    *,
    beta: float = 1.0,
    top_k: int = 3,
    weight_mode: str = "delta_tpr_weighted",
) -> np.ndarray:
    """Compute MG-EP-PT weights for one file-type expert's training data.

    Args:
        X_train_type: (n, 2568) training features (already filtered to type).
        y_train_type: (n,) labels (0/1).
        X_challenge_type: (n_ch, 2568) challenge malware features (filtered to type).
        file_type: one of {"win32", "win64", "dot_net"}.
        beta: base scaling factor.
        top_k: number of groups to use; must be 3 (default lock) for now.
        weight_mode: "uniform" (alpha_g = 1) or "delta_tpr_weighted"
            (alpha_g pre-computed from preflight numbers).

    Returns:
        weights: (n,) sample weights >= 1.0; benign rows = 1.0 exactly.
    """
    if file_type not in PE_FILE_TYPES:
        raise ValueError(f"unknown file_type {file_type!r}")
    if top_k != 3:
        # For sensitivity ablation top_K=1 or top_K=5, callers pass the right
        # `groups` list directly; here top_k=3 is the locked default.
        raise NotImplementedError(
            f"top_k={top_k} requires explicit groups override; locked default is 3"
        )
    if X_train_type.shape[1] != X_challenge_type.shape[1]:
        raise ValueError(
            f"feature dim mismatch: train {X_train_type.shape[1]} vs challenge "
            f"{X_challenge_type.shape[1]}"
        )

    groups = TOP_K_PER_TYPE[file_type]  # 3 groups
    if weight_mode == "uniform":
        alpha = {g: 1.0 for g in groups}
    elif weight_mode == "delta_tpr_weighted":
        alpha_dict = MG_EP_ALPHA_PER_TYPE_TOP3[file_type]
        alpha = {g: float(alpha_dict[g]) for g in groups}
    else:
        raise ValueError(f"unknown weight_mode {weight_mode!r}")

    centroids = compute_centroids_for_type(X_challenge_type, groups)

    n = X_train_type.shape[0]
    accum = np.zeros(n, dtype=np.float64)
    for g in groups:
        idx = indices_for_group(g)
        sims = _cosine_similarity_clipped(X_train_type[:, idx], centroids[g])
        accum += alpha[g] * sims

    is_malicious = (y_train_type.astype(bool))
    weights = np.ones(n, dtype=np.float64)
    weights[is_malicious] += beta * accum[is_malicious]
    return weights


def normalize_weights_to_unit_mean(weights: np.ndarray) -> np.ndarray:
    """Rescale so mean == 1.0 (preserves total effective sample size).

    Common practice when combining multiple weighting schemes (MG-EP × LS × SAR)
    to avoid double-counting the magnitude.
    """
    m = weights.mean()
    if m <= 0:
        raise ValueError("non-positive mean weight; cannot normalize")
    return weights / m


def clip_weights(weights: np.ndarray, R: float = 10.0) -> np.ndarray:
    """Clip weights to [1/R, R] after combining schemes (config 12 R-sensitivity)."""
    if R <= 1.0:
        raise ValueError("clip ratio R must be > 1")
    return np.clip(weights, 1.0 / R, R)
