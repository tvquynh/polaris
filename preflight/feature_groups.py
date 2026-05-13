"""Feature group indexing helpers (vendored from P0Y feature_groups.py).

Wraps ember_v3_schema (located in p01 parent dir or P0Y dir, on sys.path).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add p01 parent dir to sys.path so `import ember_v3_schema` resolves.
_THIS = Path(__file__).resolve().parent
_P01 = _THIS.parent  # p01_fte_flagship/
if str(_P01) not in sys.path:
    sys.path.insert(0, str(_P01))

from ember_v3_schema import (  # noqa: E402
    FEATURE_GROUPS, TOTAL_DIMS, CATEGORICAL_FEATURES, group_slice,
)

GROUP_CODES = list(FEATURE_GROUPS.keys())


def indices_for_group(code: str) -> np.ndarray:
    g = FEATURE_GROUPS[code]
    return np.arange(g["start"], g["end"], dtype=np.int64)


def indices_excluding_group(code: str) -> np.ndarray:
    drop = set(indices_for_group(code).tolist())
    return np.array([i for i in range(TOTAL_DIMS) if i not in drop], dtype=np.int64)


def categorical_features_after_drop(kept_indices: np.ndarray) -> list[int]:
    """Re-map CATEGORICAL_FEATURES into new indices after dropping some columns.

    LightGBM expects categorical_feature as positional indices in the *new*
    matrix, not the original schema. After we slice X[:, kept_indices], the
    categorical features that survived must be re-indexed.
    """
    kept = set(kept_indices.tolist())
    new_index_of = {old: new for new, old in enumerate(kept_indices.tolist())}
    new_cats = []
    for c in CATEGORICAL_FEATURES:
        if c in kept:
            new_cats.append(new_index_of[c])
    return new_cats


if len(GROUP_CODES) != 12:
    raise RuntimeError(f"Expected 12 groups, got {len(GROUP_CODES)}: {GROUP_CODES}")
