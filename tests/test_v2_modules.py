"""Smoke tests for P01 v2 scaffolded modules: config_v2 + multi_group_ep_per_type.

Run from p01_fte_flagship folder:
    python -m pytest tests/test_v2_modules.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_THIS = Path(__file__).resolve().parent
_P01 = _THIS.parent
_PREFLIGHT = _P01 / "preflight_per_type_lofo"
for p in (_P01, _PREFLIGHT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config_v2 as cfg
import multi_group_ep_per_type as mgep
from feature_groups import indices_for_group


# ── config_v2 sanity ────────────────────────────────────────────────────────

def test_top_k_per_type_keys():
    assert set(cfg.TOP_K_PER_TYPE.keys()) == set(cfg.PE_FILE_TYPES)
    for ft, groups in cfg.TOP_K_PER_TYPE.items():
        assert len(groups) == 3, f"top-3 for {ft} has {len(groups)} groups"


def test_top_k_per_type_values_locked():
    # Lock these against accidental mutation — preflight 2026-05-09.
    assert cfg.TOP_K_PER_TYPE["win32"] == ["HDR", "SEC", "IMP"]
    assert cfg.TOP_K_PER_TYPE["win64"] == ["IMP", "DD", "HDR"]
    assert cfg.TOP_K_PER_TYPE["dot_net"] == ["STR", "HDR", "DD"]


def test_drop_bottom_3_per_type_keys():
    assert set(cfg.DROP_BOTTOM_3_PER_TYPE.keys()) == set(cfg.PE_FILE_TYPES)
    for ft, groups in cfg.DROP_BOTTOM_3_PER_TYPE.items():
        assert len(groups) == 3


def test_drop_bottom_8_dotnet():
    assert len(cfg.DROP_BOTTOM_8_DOTNET) == 8
    # All 8 must be valid group codes
    valid = {"GFI", "BH", "BEH", "STR", "HDR", "SEC", "IMP",
             "EXP", "DD", "RH", "AUTH", "WARN"}
    assert all(g in valid for g in cfg.DROP_BOTTOM_8_DOTNET)


def test_seeds_count_10():
    assert len(cfg.SEEDS) == 10
    assert cfg.SEEDS[:5] == [42, 123, 456, 789, 1011]


def test_alpha_normalization():
    # delta_tpr_weighted alpha must sum near 3 (top-3 preserves scale).
    for ft, alpha_dict in cfg.MG_EP_ALPHA_PER_TYPE_TOP3.items():
        s = sum(v for k, v in alpha_dict.items() if k != "_sum")
        assert 2.5 < s < 3.5, f"{ft} alpha sum {s} not near 3"


def test_baseline_b_critical_keys():
    required = ["objective", "boosting_type", "n_estimators", "num_leaves",
                "feature_pre_filter", "is_unbalance", "lambda_l2"]
    for k in required:
        assert k in cfg.BASELINE_B
    assert cfg.BASELINE_B["feature_pre_filter"] is False
    assert cfg.BASELINE_B["n_estimators"] == 500


def test_sar_constants_present():
    assert hasattr(cfg, "SAR_K_LATE_CONFIRMED")
    assert hasattr(cfg, "SAR_K_REVERTED")
    assert cfg.SAR_STATE_LATE_CONFIRMED == 3
    assert cfg.SAR_STATE_REVERTED == 4


# ── multi_group_ep_per_type ─────────────────────────────────────────────────

def _make_synthetic_data(n_train=200, n_chal=50, seed=0):
    """Random synthetic training + challenge data with 2568 features."""
    rng = np.random.default_rng(seed)
    X_tr = rng.normal(size=(n_train, 2568)).astype(np.float32)
    y_tr = rng.integers(0, 2, size=n_train).astype(np.int8)
    X_ch = rng.normal(size=(n_chal, 2568)).astype(np.float32)
    return X_tr, y_tr, X_ch


def test_compute_challenge_centroid_shape():
    _, _, X_ch = _make_synthetic_data()
    c = mgep.compute_challenge_centroid(X_ch, "STR")
    expected = indices_for_group("STR").shape[0]
    assert c.shape == (expected,)


def test_cosine_similarity_clipped_orthogonal():
    centroid = np.array([1.0, 0.0, 0.0])
    X = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # orthogonal to centroid
    sims = mgep._cosine_similarity_clipped(X, centroid)
    assert np.allclose(sims, [0.0, 0.0])


def test_cosine_similarity_clipped_identical():
    centroid = np.array([1.0, 1.0, 1.0])
    X = np.array([[2.0, 2.0, 2.0]])  # parallel to centroid
    sims = mgep._cosine_similarity_clipped(X, centroid)
    assert np.allclose(sims, [1.0])


def test_cosine_similarity_clipped_negative_clipped_to_zero():
    centroid = np.array([1.0, 1.0, 1.0])
    X = np.array([[-1.0, -1.0, -1.0]])  # anti-parallel
    sims = mgep._cosine_similarity_clipped(X, centroid)
    assert np.allclose(sims, [0.0])


def test_compute_mg_ep_weights_benign_unchanged():
    X_tr, y_tr, X_ch = _make_synthetic_data()
    weights = mgep.compute_mg_ep_weights(X_tr, y_tr, X_ch, "win32",
                                          weight_mode="uniform")
    assert weights.shape == (len(y_tr),)
    # All benign rows must have weight exactly 1.0
    assert np.allclose(weights[y_tr == 0], 1.0)


def test_compute_mg_ep_weights_malicious_geq_one():
    X_tr, y_tr, X_ch = _make_synthetic_data()
    weights = mgep.compute_mg_ep_weights(X_tr, y_tr, X_ch, "win32",
                                          weight_mode="uniform")
    # All malicious weights must be >= 1.0 (cosine sim clipped to [0,1])
    assert np.all(weights[y_tr == 1] >= 1.0)


def test_compute_mg_ep_weights_unknown_filetype():
    X_tr, y_tr, X_ch = _make_synthetic_data()
    with pytest.raises(ValueError):
        mgep.compute_mg_ep_weights(X_tr, y_tr, X_ch, "unknown_type")


def test_compute_mg_ep_weights_dim_mismatch():
    X_tr, y_tr, _ = _make_synthetic_data()
    bad_chal = np.zeros((10, 1000), dtype=np.float32)  # wrong dim
    with pytest.raises(ValueError):
        mgep.compute_mg_ep_weights(X_tr, y_tr, bad_chal, "win32")


def test_compute_mg_ep_weights_uniform_vs_delta_weighted():
    X_tr, y_tr, X_ch = _make_synthetic_data()
    w_uni = mgep.compute_mg_ep_weights(X_tr, y_tr, X_ch, "win32",
                                        weight_mode="uniform")
    w_dtw = mgep.compute_mg_ep_weights(X_tr, y_tr, X_ch, "win32",
                                        weight_mode="delta_tpr_weighted")
    # Different alpha → different weights for malicious samples
    mask = y_tr == 1
    if mask.sum() > 0:
        assert not np.allclose(w_uni[mask], w_dtw[mask])


def test_normalize_weights_to_unit_mean():
    w = np.array([1.0, 2.0, 3.0, 4.0])
    norm = mgep.normalize_weights_to_unit_mean(w)
    assert pytest.approx(norm.mean()) == 1.0


def test_clip_weights():
    w = np.array([0.05, 1.0, 5.0, 50.0])
    clipped = mgep.clip_weights(w, R=10.0)
    assert clipped.min() >= 0.1 - 1e-12
    assert clipped.max() <= 10.0 + 1e-12


def test_clip_weights_invalid_R():
    with pytest.raises(ValueError):
        mgep.clip_weights(np.array([1.0]), R=0.5)
