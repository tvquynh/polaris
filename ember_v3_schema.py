#!/usr/bin/env python3
"""
EMBER2024 v3 Feature Schema — canonical source of truth.

Total dimensions: 2568
12 disjoint feature groups with EXACT half-open slices [start, end):

  Code               Name                         Python Slice   Dim
  GFI    General File Info                        [0,     7)       7
  BH     Byte Histogram                           [7,   263)     256
  BEH    Byte-Entropy Histogram                   [263, 519)     256
  STR    Strings                                  [519, 696)     177
  HDR    PE Header                                [696, 770)      74
  SEC    Section Information                      [770, 994)     224
  IMP    Imports                                  [994, 2276)  1,282
  EXP    Exports                                  [2276, 2405)   129
  DD     Data Directories                         [2405, 2439)    34
  RH     Rich Header                              [2439, 2472)    33
  AUTH   Authenticode Signature                   [2472, 2480)     8
  WARN   PE Format Warnings                       [2480, 2568)    88

Usage:
    from ember_v3_schema import FEATURE_GROUPS, group_slice, SIZE_FEATURE_IDX
    beh = X[:, group_slice('BEH')]       # -> shape (N, 256)

Rules:
    - KHONG hardcode indices trong code moi. Import module nay.
    - Defensive check: assert_schema(X) truoc khi slice.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import numpy as np

# ── Canonical 12-group schema (half-open intervals) ──────────────────────────
FEATURE_GROUPS: "OrderedDict[str, Dict]" = OrderedDict([
    ("GFI",  {"name": "General File Info",       "start": 0,    "end": 7,    "dim": 7}),
    ("BH",   {"name": "Byte Histogram",          "start": 7,    "end": 263,  "dim": 256}),
    ("BEH",  {"name": "Byte-Entropy Histogram",  "start": 263,  "end": 519,  "dim": 256}),
    ("STR",  {"name": "Strings",                 "start": 519,  "end": 696,  "dim": 177}),
    ("HDR",  {"name": "PE Header",               "start": 696,  "end": 770,  "dim": 74}),
    ("SEC",  {"name": "Section Information",     "start": 770,  "end": 994,  "dim": 224}),
    ("IMP",  {"name": "Imports",                 "start": 994,  "end": 2276, "dim": 1282}),
    ("EXP",  {"name": "Exports",                 "start": 2276, "end": 2405, "dim": 129}),
    ("DD",   {"name": "Data Directories",        "start": 2405, "end": 2439, "dim": 34}),
    ("RH",   {"name": "Rich Header",             "start": 2439, "end": 2472, "dim": 33}),
    ("AUTH", {"name": "Authenticode Signature",  "start": 2472, "end": 2480, "dim": 8}),
    ("WARN", {"name": "PE Format Warnings",      "start": 2480, "end": 2568, "dim": 88}),
])

TOTAL_DIMS = 2568
SIZE_FEATURE_IDX = 0  # feature_0000 = raw file size (bytes) in GFI group

# Categorical feature indices — match thrember/model.py (Joyce et al. KDD'25):
#   train_model() passes `categorical_feature=[2, 3, 4, 5, 6, 701, 702]`
#   to lgb.Dataset at lines 366-367, 404-405.
#
#   Indices 2-6 (within GFI [0, 7)):
#       first bytes used to infer file type — discrete format magic values,
#       NOT numeric; tree should treat as categorical.
#   Indices 701, 702 (within HDR [696, 770), offsets 5 and 6):
#       PE machine type + subsystem — enumerated PE/COFF codes.
#
# LightGBM uses different (one-hot style) splits for categorical features,
# which matters for reproducing the author's baseline exactly.
CATEGORICAL_FEATURES = [2, 3, 4, 5, 6, 701, 702]


def group_slice(code: str) -> slice:
    """Return Python slice object for a feature group."""
    g = FEATURE_GROUPS[code]
    return slice(g["start"], g["end"])


def group_indices(code: str) -> np.ndarray:
    """Return int array of indices for a feature group."""
    g = FEATURE_GROUPS[code]
    return np.arange(g["start"], g["end"], dtype=np.int64)


def assert_schema(X: np.ndarray) -> None:
    """Defensive check: X must have exactly TOTAL_DIMS columns."""
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    if X.shape[1] != TOTAL_DIMS:
        raise ValueError(
            f"Feature dim mismatch: expected {TOTAL_DIMS}, got {X.shape[1]}. "
            f"EMBER2024 v3 parquet must yield 2568-dim vectors."
        )
    # Sanity: sum of group dims equals total
    total = sum(g["dim"] for g in FEATURE_GROUPS.values())
    assert total == TOTAL_DIMS, (
        f"Schema error: group dims sum to {total}, expected {TOTAL_DIMS}"
    )


def describe() -> str:
    """Human-readable schema table."""
    lines = [
        f"{'Code':<6}{'Name':<30}{'Slice':<15}{'Dim':>6}",
        "-" * 57,
    ]
    for code, g in FEATURE_GROUPS.items():
        sl = f"[{g['start']}, {g['end']})"
        lines.append(f"{code:<6}{g['name']:<30}{sl:<15}{g['dim']:>6}")
    lines.append("-" * 57)
    lines.append(f"{'TOTAL':<36}{'':<15}{TOTAL_DIMS:>6}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    print()
    # Self-test
    X_fake = np.zeros((2, TOTAL_DIMS), dtype=np.float32)
    assert_schema(X_fake)
    assert X_fake[:, group_slice("BEH")].shape == (2, 256)
    assert X_fake[:, group_slice("IMP")].shape == (2, 1282)
    print("[OK] Schema self-test passed (BEH=256d, IMP=1282d).")
