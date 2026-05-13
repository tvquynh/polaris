"""Smoke tests for P01 v2 SAR + PETO modules.

Run: python -m pytest tests/test_sar_peto.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_THIS = Path(__file__).resolve().parent
_P01 = _THIS.parent
if str(_P01) not in sys.path:
    sys.path.insert(0, str(_P01))

import state_aware_reweighting as sar
import peto


# ── SAR ─────────────────────────────────────────────────────────────────────

def test_sar_weights_state_0_or_1_returns_one():
    states = np.array([0, 1, 0, 1, -1])  # all stable_benign / stable_malicious / NA
    w = sar.compute_sar_weights(states)
    assert np.allclose(w, 1.0)


def test_sar_weights_state_3_boost():
    states = np.array([3, 3, 0, 1])
    w = sar.compute_sar_weights(states, k_late_confirmed=2.0, k_reverted=0.5)
    expected = np.array([3.0, 3.0, 1.0, 1.0])  # state 3 → 1+2=3
    assert np.allclose(w, expected)


def test_sar_weights_state_4_boost():
    states = np.array([4, 4, 0])
    w = sar.compute_sar_weights(states, k_late_confirmed=1.0, k_reverted=0.7)
    expected = np.array([1.7, 1.7, 1.0])
    assert np.allclose(w, expected)


def test_sar_weights_state_3_and_4_independent():
    states = np.array([0, 1, 2, 3, 4, -1])
    w = sar.compute_sar_weights(states, k_late_confirmed=1.5, k_reverted=0.3)
    expected = np.array([1.0, 1.0, 1.0, 2.5, 1.3, 1.0])
    assert np.allclose(w, expected)


def test_sar_weights_negative_k_raises():
    with pytest.raises(ValueError):
        sar.compute_sar_weights(np.array([0]), k_late_confirmed=-1.0)
    with pytest.raises(ValueError):
        sar.compute_sar_weights(np.array([0]), k_reverted=-0.5)


def test_sar_weights_shape_preserved():
    states = np.random.default_rng(0).integers(-1, 5, size=100)
    w = sar.compute_sar_weights(states)
    assert w.shape == (100,)
    assert np.all(w >= 1.0)


def test_state_distribution_summary_keys():
    states = np.array([0, 0, 1, 2, 3, 3, 4, -1])
    s = sar.state_distribution_summary(states)
    assert s["stable_benign"] == 2
    assert s["stable_malicious"] == 1
    assert s["grayware"] == 1
    assert s["late_confirmed_malicious"] == 2
    assert s["reverted_or_disputed"] == 1
    assert s["NA"] == 1


def test_load_state_labels_missing_file():
    arr = np.array(["abc", "def"])
    with pytest.raises(FileNotFoundError):
        sar.load_state_labels_for_sha256(arr, parquet_path="/nonexistent/path.parquet")


# ── PETO ────────────────────────────────────────────────────────────────────

def _make_synthetic_scores(rng_seed=42):
    """Three file types with separable benign vs malware score distributions."""
    rng = np.random.default_rng(rng_seed)
    benign = {
        "win32":   rng.uniform(0, 0.3, size=2000),
        "win64":   rng.uniform(0, 0.3, size=1000),
        "dot_net": rng.uniform(0, 0.3, size=500),
    }
    malware = {
        "win32":   rng.uniform(0.5, 0.95, size=300),
        "win64":   rng.uniform(0.5, 0.95, size=200),
        "dot_net": rng.uniform(0.6, 0.99, size=100),
    }
    return benign, malware


def test_peto_returns_3_thresholds():
    b, m = _make_synthetic_scores()
    thresholds = peto.compute_peto_thresholds(b, m, fpr_budget=0.01, n_grid=50)
    assert set(thresholds.keys()) == {"win32", "win64", "dot_net"}
    for t, theta in thresholds.items():
        assert 0.0 <= theta <= 1.0


def test_peto_satisfies_fpr_budget():
    b, m = _make_synthetic_scores()
    fpr_budget = 0.01
    thresholds = peto.compute_peto_thresholds(b, m, fpr_budget=fpr_budget, n_grid=50)
    diag = peto.evaluate_peto(thresholds, b, m)
    # Must be within budget (with small tolerance for grid quantization)
    assert diag["ensemble_fpr"] <= fpr_budget + 1e-6


def test_peto_high_tpr_with_separable_data():
    b, m = _make_synthetic_scores()
    thresholds = peto.compute_peto_thresholds(b, m, fpr_budget=0.01, n_grid=50)
    diag = peto.evaluate_peto(thresholds, b, m)
    # Synthetic data is well-separable; ensemble TPR should be high
    assert diag["ensemble_tpr"] > 0.8


def test_peto_invalid_n_types():
    b = {"win32": np.array([0.1, 0.2]), "win64": np.array([0.1, 0.2])}  # only 2
    m = {"win32": np.array([0.5, 0.6]), "win64": np.array([0.5, 0.6])}
    with pytest.raises(NotImplementedError):
        peto.compute_peto_thresholds(b, m)


def test_peto_keys_mismatch():
    b = {"win32": np.array([0.1]), "win64": np.array([0.1]), "dot_net": np.array([0.1])}
    m = {"win32": np.array([0.5]), "win64": np.array([0.5]), "elf": np.array([0.5])}  # mismatch
    with pytest.raises(ValueError):
        peto.compute_peto_thresholds(b, m)


def test_peto_empty_scores():
    b = {"win32": np.array([]), "win64": np.array([0.1]), "dot_net": np.array([0.1])}
    m = {"win32": np.array([0.5]), "win64": np.array([0.5]), "dot_net": np.array([0.5])}
    with pytest.raises(ValueError):
        peto.compute_peto_thresholds(b, m)


def test_peto_evaluate_diagnostic_keys():
    b, m = _make_synthetic_scores()
    diag = peto.evaluate_peto({"win32": 0.5, "win64": 0.5, "dot_net": 0.5}, b, m)
    for prefix in ("tpr_", "fpr_", "threshold_"):
        for ft in ("win32", "win64", "dot_net"):
            assert f"{prefix}{ft}" in diag
    assert "ensemble_fpr" in diag
    assert "ensemble_tpr" in diag


def test_peto_finds_better_than_uniform_threshold():
    """Per-expert thresholds must beat a single global threshold under same FPR budget."""
    b, m = _make_synthetic_scores()
    fpr_budget = 0.01
    # Per-expert PETO
    pe_thresholds = peto.compute_peto_thresholds(b, m, fpr_budget=fpr_budget, n_grid=100)
    pe_diag = peto.evaluate_peto(pe_thresholds, b, m)
    # Global threshold (single theta for all)
    all_b = np.concatenate(list(b.values()))
    global_theta = float(np.quantile(all_b, 1.0 - fpr_budget, method="higher"))
    global_thresholds = {t: global_theta for t in b}
    g_diag = peto.evaluate_peto(global_thresholds, b, m)
    # PETO should not be worse than global on TPR (typically better; equal in degenerate cases)
    assert pe_diag["ensemble_tpr"] >= g_diag["ensemble_tpr"] - 1e-6
