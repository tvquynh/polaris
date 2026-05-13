"""state_aware_reweighting.py — SAR component for P01 v2.

Replaces original CSC (Covariate-Shift Correction) because P11+ challenge
samples have state5 = -1 (protected NA, not relabeled by P11+ design — they
are the official MLSec challenge ground truth, not subject to VT refresh).

Approximation: challenge samples behave like state 3 (late_confirmed_malicious)
+ state 4 (reverted_or_disputed) per P11+ semantic definition. Both states
represent "VT consensus shift" — challenge set members are samples that were
initially missed and later confirmed malicious, matching state 3 most closely.

Formula (Eq. 2 P01 v2):
    w_sar_i = 1 + k_3 * I[state_i = 3] + k_4 * I[state_i = 4]

where state_i comes from P11+ relabel_frame.parquet column `state5_tau5`,
joined to training samples via sha256.

Both k_3 and k_4 are Optuna-tunable in HPO.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from config_v2 import (  # noqa: E402
    SAR_K_LATE_CONFIRMED,
    SAR_K_REVERTED,
    SAR_STATE_LATE_CONFIRMED,
    SAR_STATE_REVERTED,
    P11_RELABEL_FRAME_PATH,
    P11_STATE_COLUMN,
    P11_SHA256_COLUMN,
    P11_FILETYPE_COLUMN,
)


def compute_sar_weights(
    state_train: np.ndarray,
    *,
    k_late_confirmed: float = SAR_K_LATE_CONFIRMED,
    k_reverted: float = SAR_K_REVERTED,
) -> np.ndarray:
    """Compute SAR weights per training sample.

    Args:
        state_train: (n,) int array of state5_tau5 values per training sample.
            Codes: 0=stable_benign, 1=stable_malicious, 2=grayware,
            3=late_confirmed_malicious, 4=reverted_or_disputed, -1=NA.
        k_late_confirmed: boost for state 3 (Optuna-tunable).
        k_reverted: boost for state 4 (Optuna-tunable, default smaller).

    Returns:
        weights: (n,) sample weights >= 1.0.
    """
    if k_late_confirmed < 0 or k_reverted < 0:
        raise ValueError("k weights must be non-negative")
    state = np.asarray(state_train, dtype=np.int64)
    weights = np.ones(state.shape[0], dtype=np.float64)
    weights[state == SAR_STATE_LATE_CONFIRMED] += float(k_late_confirmed)
    weights[state == SAR_STATE_REVERTED] += float(k_reverted)
    return weights


def load_state_labels_for_sha256(
    sha256_array: np.ndarray,
    *,
    parquet_path: Optional[str] = None,
    state_column: str = P11_STATE_COLUMN,
) -> np.ndarray:
    """Load state5 labels from P11+ relabel_frame.parquet, aligned to sha256 order.

    Args:
        sha256_array: (n,) array of sha256 strings (training set order).
        parquet_path: path to relabel_frame.parquet. Defaults to P11_RELABEL_FRAME_PATH
            relative to this file's parent directory.
        state_column: which state column to use (default state5_tau5 = paper main).

    Returns:
        states: (n,) int8 array of state codes; -1 for samples not found in P11+.
    """
    import pyarrow.parquet as pq

    if parquet_path is None:
        parquet_path = str((_THIS / P11_RELABEL_FRAME_PATH).resolve())
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"P11+ relabel_frame not found: {p}")

    table = pq.read_table(
        p, columns=[P11_SHA256_COLUMN, state_column],
    )
    sha_to_state: Dict[str, int] = {}
    sha_col = table.column(P11_SHA256_COLUMN).to_pylist()
    state_col = table.column(state_column).to_pylist()
    for sha, st in zip(sha_col, state_col):
        if st is None:
            sha_to_state[sha] = -1
        else:
            sha_to_state[sha] = int(st)

    states = np.full(len(sha256_array), -1, dtype=np.int64)
    for i, sha in enumerate(sha256_array):
        if sha in sha_to_state:
            states[i] = sha_to_state[sha]
    return states


def state_distribution_summary(states: np.ndarray) -> Dict[str, int]:
    """Diagnostic — counts per state code.

    Returns dict with semantic keys for readability.
    """
    state_names = {
        -1: "NA",
        0: "stable_benign",
        1: "stable_malicious",
        2: "grayware",
        3: "late_confirmed_malicious",
        4: "reverted_or_disputed",
    }
    out = {name: 0 for name in state_names.values()}
    unique, counts = np.unique(states, return_counts=True)
    for u, c in zip(unique, counts):
        name = state_names.get(int(u), f"unknown_{int(u)}")
        out[name] = int(c)
    return out
