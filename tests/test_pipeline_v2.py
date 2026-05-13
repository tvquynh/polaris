"""Smoke tests for pipeline_v2.py (no real data — synthetic + mocks).

Run: python -m pytest tests/test_pipeline_v2.py -v
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

import pipeline_v2 as pv2
import config_v2 as cfg


def test_config_registry_size_and_required_cells():
    assert "baseline_B" in pv2.CONFIG_REGISTRY
    assert "proposed_v2" in pv2.CONFIG_REGISTRY
    assert "fte_ls" in pv2.CONFIG_REGISTRY  # paper-Feb-2026 ablation
    assert "fte_beh_only" in pv2.CONFIG_REGISTRY  # legacy v1 contrast
    assert len(pv2.CONFIG_REGISTRY) >= 14


def test_proposed_v2_has_all_4_components():
    kw = pv2.CONFIG_REGISTRY["proposed_v2"]
    assert kw.get("use_fte") is True
    assert kw.get("use_mgep") is True
    assert kw.get("use_sar") is True
    assert kw.get("use_ls") is True
    assert kw.get("use_peto") is True
    assert kw.get("use_hpo") is True


def test_compute_ls_weights_no_malware():
    file_size = np.array([100.0, 200.0, 300.0])
    y = np.zeros(3)  # all benign
    w = pv2.compute_ls_weights(file_size, y, alpha=11.0, percentile=90.0)
    assert np.allclose(w, 1.0)  # no malware → no boost


def test_compute_ls_weights_boosts_large_malware():
    file_size = np.array([100.0, 200.0, 5000.0, 50.0])  # large=5000
    y = np.array([1, 1, 1, 0])
    w = pv2.compute_ls_weights(file_size, y, alpha=11.0, percentile=50.0)
    # Threshold at P50 of malware = median(100, 200, 5000) = 200
    # Sample 2 (size 5000, mal): boost. Samples 0,1 (size 100, 200, mal): no boost.
    assert w[2] == pytest.approx(12.0)  # 1 + 11
    assert w[0] == 1.0
    assert w[1] == 1.0  # 200 NOT > 200
    assert w[3] == 1.0  # benign


def test_compute_combined_weights_all_off_returns_ones():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(50, 2568)).astype(np.float32)
    y_tr = rng.integers(0, 2, size=50)
    X_ch = rng.normal(size=(20, 2568)).astype(np.float32)
    w = pv2.compute_combined_weights(
        X_tr, y_tr, X_ch, file_size=None, states=None, file_type="win32",
        use_mgep=False, use_sar=False, use_ls=False,
    )
    assert np.allclose(w, 1.0)


def test_compute_combined_weights_only_mgep():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(50, 2568)).astype(np.float32)
    y_tr = rng.integers(0, 2, size=50)
    X_ch = rng.normal(size=(20, 2568)).astype(np.float32)
    w = pv2.compute_combined_weights(
        X_tr, y_tr, X_ch, file_size=None, states=None, file_type="win32",
        use_mgep=True, mgep_ranking="per_type", weight_clip_R=10.0,
    )
    # All benign rows must be 1.0
    assert np.allclose(w[y_tr == 0], 1.0)
    # Malicious rows >= 1.0 (cosine clipped to [0,1])
    assert np.all(w[y_tr == 1] >= 1.0)


def test_compute_combined_weights_only_sar():
    states = np.array([0, 1, 2, 3, 4, -1])
    y = np.array([0, 1, 0, 1, 1, 0])
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(6, 2568)).astype(np.float32)
    X_ch = rng.normal(size=(20, 2568)).astype(np.float32)
    w = pv2.compute_combined_weights(
        X_tr, y, X_ch, file_size=None, states=states, file_type="win32",
        use_sar=True, weight_clip_R=10.0,
    )
    # state 3 → 1 + k_3, state 4 → 1 + k_4, others → 1
    assert w[3] == pytest.approx(1.0 + cfg.SAR_K_LATE_CONFIRMED)
    assert w[4] == pytest.approx(1.0 + cfg.SAR_K_REVERTED)
    assert w[0] == 1.0


def test_compute_combined_weights_clipped():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(50, 2568)).astype(np.float32)
    y_tr = np.ones(50, dtype=np.int8)  # all malware → all get boost
    X_ch = rng.normal(size=(20, 2568)).astype(np.float32)
    states = np.full(50, 3, dtype=np.int64)  # all late_confirmed
    file_size = rng.uniform(1e6, 1e8, size=50)  # all large

    w = pv2.compute_combined_weights(
        X_tr, y_tr, X_ch, file_size=file_size, states=states, file_type="win32",
        use_mgep=True, use_sar=True, use_ls=True, ls_alpha=20.0,
        weight_clip_R=10.0,
    )
    # All clipped to [0.1, 10]
    assert w.max() <= 10.0 + 1e-9
    assert w.min() >= 0.1 - 1e-9


def test_apply_drop_bottom_per_type_3_win32():
    X_tr = np.random.default_rng(0).normal(size=(10, 2568)).astype(np.float32)
    X_te = np.random.default_rng(1).normal(size=(5, 2568)).astype(np.float32)
    X_ch = np.random.default_rng(2).normal(size=(3, 2568)).astype(np.float32)
    X_tr_r, X_te_r, X_ch_r, cat_r = pv2.apply_drop_bottom(
        X_tr, X_te, X_ch, "win32", "per_type_3",
    )
    # Win32 drops GFI(7) + BH(256) + EXP(129) = 392 features
    expected = 2568 - (7 + 256 + 129)
    assert X_tr_r.shape[1] == expected
    assert X_te_r.shape[1] == expected
    assert X_ch_r.shape[1] == expected
    assert all(c < expected for c in cat_r)


def test_apply_drop_bottom_dotnet_8_only_dotnet():
    X_tr = np.random.default_rng(0).normal(size=(10, 2568)).astype(np.float32)
    X_te = np.random.default_rng(1).normal(size=(5, 2568)).astype(np.float32)
    X_ch = np.random.default_rng(2).normal(size=(3, 2568)).astype(np.float32)
    # For win32, dotnet_8 should NOT drop anything
    X_tr_r, _, _, _ = pv2.apply_drop_bottom(X_tr, X_te, X_ch, "win32", "dotnet_8")
    assert X_tr_r.shape[1] == 2568  # unchanged
    # For dot_net, drop bottom 8 = 8 groups (BH/GFI/EXP/AUTH/IMP/SEC/BEH/RH)
    X_tr_r, _, _, _ = pv2.apply_drop_bottom(X_tr, X_te, X_ch, "dot_net", "dotnet_8")
    # 8 groups removed → fewer features
    assert X_tr_r.shape[1] < 2568


def test_threshold_for_fpr_basic():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # FPR=0.1 → top 10% benign exceeds threshold; threshold ≈ 0.9
    t = pv2.threshold_for_fpr(scores, 0.1)
    fpr_actual = (scores >= t).mean()
    assert fpr_actual <= 0.1 + 1e-9
    assert fpr_actual >= 0.05  # some samples must exceed


def test_compute_metrics_simple_separable():
    # Highly separable benign (low scores) vs malware (high scores)
    sb = {"win32": np.array([0.01, 0.02, 0.05, 0.08])}
    sm = {"win32": np.array([0.95, 0.97, 0.99])}
    thresholds = {"win32": 0.5}
    m = pv2.compute_metrics(sb, sm, thresholds)
    assert m["per_type"]["win32"]["tpr_at_fpr_0.01_challenge"] == 1.0  # all mal > 0.5
    assert m["per_type"]["win32"]["fpr_at_threshold"] == 0.0
    assert m["ensemble"]["ensemble_tpr_at_fpr_0.01"] == 1.0
