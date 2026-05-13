"""perturbation_robustness.py — Synthetic feature perturbation for adversarial robustness analysis.

Simulate attacker manipulating specific feature groups on challenge samples.
For each perturbation type, measure TPR drop:
    - "size_drop"  — reduce file_size feature (LS evade) — simulate code shrinkage
    - "str_zero"   — zero out STR features — simulate string padding/obfuscation
    - "hdr_noise"  — add noise to HDR features — simulate header manipulation
    - "sec_zero"   — zero out SEC features — simulate benign section injection
    - "all_cosmetic" — combine STR + HDR + SEC (cosmetic but multi-feature)

For each model + perturbation, compare TPR drop magnitude.
Hypothesis: simpler models (fte_ls relies on size) drop more on size_drop;
complex models (4-component) drop less but require multi-group attack.

Outputs:
    perturbation_robustness.json
    perturbation_robustness.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import lightgbm as lgb

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS / "preflight_per_type_lofo"))
from feature_groups import indices_for_group
sys.path.insert(0, str(_THIS))
from ember_v3_schema import SIZE_FEATURE_IDX
from ember2024_parquet_loader import load_challenge_mixed


def threshold_for_fpr(scores_benign: np.ndarray, fpr: float) -> float:
    return float(np.quantile(scores_benign, 1.0 - fpr, method="higher"))


def predict(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        return np.array([])
    return booster.predict(X, num_iteration=booster.best_iteration).astype(np.float64)


PERTURBATIONS = {
    "no_attack": None,  # baseline (no perturbation)
    "size_drop_50pct": ("size_drop", 0.5),
    "size_drop_20pct": ("size_drop", 0.2),
    "str_zero": ("group_zero", "STR"),
    "hdr_zero": ("group_zero", "HDR"),
    "sec_zero": ("group_zero", "SEC"),
    "str_hdr_sec_zero": ("multi_group_zero", ["STR", "HDR", "SEC"]),
    "imp_zero": ("group_zero", "IMP"),
}


def apply_perturbation(X: np.ndarray, perturbation_type: str, param) -> np.ndarray:
    """Apply perturbation to feature matrix. Returns new array (X not modified)."""
    X_new = X.copy().astype(np.float64)
    if perturbation_type == "size_drop":
        # Multiply file_size by (1 - param) — simulate attacker shrinking malware
        X_new[:, SIZE_FEATURE_IDX] *= (1.0 - param)
    elif perturbation_type == "group_zero":
        idx = indices_for_group(param)
        idx = idx[idx < X.shape[1]]
        X_new[:, idx] = 0.0
    elif perturbation_type == "multi_group_zero":
        for grp in param:
            idx = indices_for_group(grp)
            idx = idx[idx < X.shape[1]]
            X_new[:, idx] = 0.0
    return X_new


def evaluate_robustness(models_dir: Path, parquet_dir: Path) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    # Per file type evaluation
    for ft in ["win32", "win64", "dot_net"]:
        X_te_b, _, X_ch, _, _, _ = load_challenge_mixed(str(parquet_dir), [ft])

        for cfg_dir in sorted(models_dir.iterdir()):
            if not cfg_dir.is_dir():
                continue
            cfg = cfg_dir.name
            # Look for model
            seed_dir = cfg_dir / "seed_42"
            unified = seed_dir / "model_unified.txt"
            per_type = seed_dir / f"model_{ft}.txt"
            if unified.exists():
                model_path = unified
            elif per_type.exists():
                model_path = per_type
            else:
                continue

            booster = lgb.Booster(model_file=str(model_path))

            # Baseline (no perturbation): compute TPR @ 1%FPR
            scores_b = predict(booster, X_te_b)
            scores_m = predict(booster, X_ch)
            if scores_b.size == 0 or scores_m.size == 0:
                continue
            theta = threshold_for_fpr(scores_b, 0.01)
            tpr_baseline = float(np.mean(scores_m >= theta))

            cfg_key = f"{cfg}__{ft}"
            results[cfg_key] = {"config": cfg, "file_type": ft,
                                "tpr_at_1pct_fpr_baseline": tpr_baseline,
                                "perturbations": {}}

            # Apply each perturbation
            for name, params in PERTURBATIONS.items():
                if params is None:
                    continue  # skip no_attack (already baseline)
                ptype, pparam = params
                X_ch_pert = apply_perturbation(X_ch, ptype, pparam)
                scores_m_pert = predict(booster, X_ch_pert)
                tpr_pert = float(np.mean(scores_m_pert >= theta))
                drop_pp = (tpr_baseline - tpr_pert) * 100
                results[cfg_key]["perturbations"][name] = {
                    "tpr_after_attack": tpr_pert,
                    "tpr_drop_pp": drop_pp,
                }

    return results


def render_md(results: Dict[str, Any]) -> str:
    L = []
    L.append("# Perturbation Robustness Analysis\n")
    L.append("TPR drop (pp) when attacker manipulates specific features on challenge samples.")
    L.append("Higher drop = model more vulnerable to that attack vector.\n")

    # Build wide table
    attacks = ["size_drop_50pct", "size_drop_20pct", "str_zero", "hdr_zero",
               "sec_zero", "str_hdr_sec_zero", "imp_zero"]
    L.append("\n| Config / Type | TPR base | " + " | ".join(attacks) + " |")
    L.append("|:---|---:|" + "---:|" * len(attacks))

    for key, data in results.items():
        cells = [f"{data['tpr_at_1pct_fpr_baseline']:.4f}"]
        for atk in attacks:
            pert = data["perturbations"].get(atk)
            if pert is None:
                cells.append("—")
            else:
                cells.append(f"-{pert['tpr_drop_pp']:.2f}pp")
        L.append(f"| `{key}` | " + " | ".join(cells) + " |")

    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_robustness(models_dir, Path(args.parquet_dir))

    (out_dir / "perturbation_robustness.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "perturbation_robustness.md").write_text(render_md(results), encoding="utf-8")
    print(f"Wrote: {out_dir / 'perturbation_robustness.json'}")
    print(f"Wrote: {out_dir / 'perturbation_robustness.md'}")


if __name__ == "__main__":
    main()
