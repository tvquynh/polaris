"""feature_importance_analysis.py — Defense-in-depth analysis via feature importance dispersion.

For each config's saved LightGBM models (1 per file type for FTE, 1 unified for baseline),
compute:
    - Per-feature gain importance
    - Dispersion metrics:
        * Top-10/50/100 cumulative gain share
        * Gini coefficient (high = concentrated, low = spread)
        * Effective features (gain > 1% of total)
    - Per-group importance (sum gain over each feature group)

Hypothesis: simpler methods (fte_ls) concentrate importance on fewer features
(higher Gini, lower effective N) → easier attack surface. Complex methods (4-component)
spread importance → larger attack surface required.

Outputs:
    importance_dispersion.json
    importance_dispersion.md
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
from feature_groups import GROUP_CODES, indices_for_group


def gini(values: np.ndarray) -> float:
    """Gini coefficient of distribution (0=equal, 1=max concentrated)."""
    if values.size == 0 or values.sum() == 0:
        return 0.0
    sorted_v = np.sort(values)
    n = len(values)
    cum = np.cumsum(sorted_v)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def analyze_booster(model_path: Path) -> Dict[str, Any]:
    """Load LightGBM booster + compute feature importance dispersion."""
    booster = lgb.Booster(model_file=str(model_path))
    imp_gain = np.array(booster.feature_importance(importance_type="gain"), dtype=np.float64)
    total = imp_gain.sum()
    if total == 0:
        return {"error": "all-zero importance"}

    # Normalize to share of total
    share = imp_gain / total

    # Sort descending
    sorted_desc = np.sort(share)[::-1]

    # Cumulative shares at top-K
    cum10 = float(sorted_desc[:10].sum())
    cum50 = float(sorted_desc[:50].sum())
    cum100 = float(sorted_desc[:100].sum())

    # Effective features (share > 1%)
    n_effective_1pct = int((share > 0.01).sum())
    n_effective_01pct = int((share > 0.001).sum())
    n_nonzero = int((share > 0).sum())

    # Gini coefficient
    g = gini(imp_gain)

    # Per-group total share
    group_share = {}
    for grp in GROUP_CODES:
        idx = indices_for_group(grp)
        if idx.max() < len(share):  # only if features exist
            group_share[grp] = float(share[idx[idx < len(share)]].sum())
        else:
            group_share[grp] = 0.0

    return {
        "n_features_total": int(len(share)),
        "n_features_nonzero": n_nonzero,
        "n_features_above_1pct": n_effective_1pct,
        "n_features_above_01pct": n_effective_01pct,
        "top10_cumulative_share": cum10,
        "top50_cumulative_share": cum50,
        "top100_cumulative_share": cum100,
        "gini_coefficient": g,
        "per_group_share": group_share,
    }


def analyze_config(config_dir: Path) -> Dict[str, Any]:
    """Analyze all booster files in a config/seed_42 dir."""
    seed_dir = config_dir / "seed_42"
    out: Dict[str, Any] = {"config": config_dir.name, "models": {}}

    # Try unified model first (baseline)
    unified = seed_dir / "model_unified.txt"
    if unified.exists():
        out["models"]["unified"] = analyze_booster(unified)
        return out

    # Otherwise FTE per-type
    for ft in ["win32", "win64", "dot_net"]:
        mp = seed_dir / f"model_{ft}.txt"
        if mp.exists():
            out["models"][ft] = analyze_booster(mp)
    return out


def render_md(results: Dict[str, Any]) -> str:
    L = []
    L.append("# Feature Importance Dispersion Analysis (Defense-in-Depth)\n")
    L.append("Larger Gini + smaller `n_above_1pct` = more concentrated risk")
    L.append("(easier single-feature attack). Apply NCS's defense-in-depth hypothesis.\n")

    L.append("\n## Summary table — top dispersion metrics\n")
    L.append("| Config | Model | Gini | N feat ≥1% | N feat ≥0.1% | Top-10 share | Top-50 share |")
    L.append("|:---|:---|---:|---:|---:|---:|---:|")
    for cfg_name, data in results.items():
        for model_name, m in data["models"].items():
            if "error" in m:
                continue
            L.append(f"| `{cfg_name}` | {model_name} | {m['gini_coefficient']:.4f} | "
                     f"{m['n_features_above_1pct']} | {m['n_features_above_01pct']} | "
                     f"{m['top10_cumulative_share']:.3f} | {m['top50_cumulative_share']:.3f} |")

    L.append("\n## Per-group importance share (sum of feature gains in each group)\n")
    L.append("| Config / Model | GFI | BH | BEH | STR | HDR | SEC | IMP | EXP | DD | RH | AUTH | WARN |")
    L.append("|:---|" + "---:|" * 12)
    for cfg_name, data in results.items():
        for model_name, m in data["models"].items():
            if "error" in m:
                continue
            cells = [f"{m['per_group_share'].get(g, 0):.3f}" for g in
                     ["GFI", "BH", "BEH", "STR", "HDR", "SEC", "IMP", "EXP", "DD", "RH", "AUTH", "WARN"]]
            L.append(f"| `{cfg_name}` / {model_name} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True,
                    help="Dir containing <config>/seed_42/model_*.txt files")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for cfg_dir in sorted(models_dir.iterdir()):
        if not cfg_dir.is_dir():
            continue
        if not (cfg_dir / "seed_42").exists():
            continue
        print(f"Analyzing {cfg_dir.name} ...")
        results[cfg_dir.name] = analyze_config(cfg_dir)

    (out_dir / "importance_dispersion.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "importance_dispersion.md").write_text(render_md(results), encoding="utf-8")
    print(f"Wrote: {out_dir / 'importance_dispersion.json'}")
    print(f"Wrote: {out_dir / 'importance_dispersion.md'}")


if __name__ == "__main__":
    main()
