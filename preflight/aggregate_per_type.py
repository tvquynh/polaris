#!/usr/bin/env python3
"""
aggregate_per_type.py — cross-seed/cross-type ranking for preflight.

Aggregates per-type group LOGO across 5 seeds → produces:
- top_k_per_type.json (machine-readable, source of truth for P01 v2 MG-EP-PT)
- top_k_per_type.md (human-readable summary)
- consistency check: Kendall tau global vs per-type
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


PE_FILE_TYPES = ["win32", "win64", "dot_net"]
GROUP_CODES = ["GFI", "BH", "BEH", "STR", "HDR", "SEC", "IMP", "EXP", "DD", "RH", "AUTH", "WARN"]
SEEDS = [42, 123, 456, 789, 1011, 2026, 3141, 4242, 5555, 6789]
PRIMARY_METRIC = "tpr_at_fpr_001_challenge"


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def aggregate_one_type(results_dir: Path, ft: str, metric: str = PRIMARY_METRIC) -> dict:
    full_per_seed: Dict[int, float] = {}
    logo_per_seed: Dict[str, Dict[int, float]] = {g: {} for g in GROUP_CODES}
    for s in SEEDS:
        full_path = results_dir / f"seed_{s}" / f"full_metrics_{ft}.json"
        grp_path = results_dir / f"seed_{s}" / f"group_metrics_{ft}.json"
        if not full_path.exists() or not grp_path.exists():
            raise FileNotFoundError(f"Missing {ft} files for seed {s} under {results_dir}")
        full_d = json.loads(full_path.read_text())
        grp_d = json.loads(grp_path.read_text())
        full_per_seed[s] = float(full_d["metrics"][metric])
        for g in GROUP_CODES:
            if g not in grp_d:
                raise KeyError(f"Group {g} missing in seed {s} {ft}")
            logo_per_seed[g][s] = float(grp_d[g]["metrics"][metric])

    rng = np.random.default_rng(42)
    results = []
    for g in GROUP_CODES:
        deltas = np.array([full_per_seed[s] - logo_per_seed[g][s] for s in SEEDS], dtype=np.float64)
        mean_d = float(deltas.mean())
        std_d = float(deltas.std(ddof=1))
        ci_lo, ci_hi = bootstrap_ci(deltas, rng=rng)
        try:
            _, pval = stats.wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
            pval = float(pval)
        except ValueError:
            pval = 1.0
        results.append({
            "group": g,
            "mean_delta_tpr": mean_d,
            "std": std_d,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "wilcoxon_p_raw": pval,
            "deltas_per_seed": deltas.tolist(),
        })

    pvals = np.array([r["wilcoxon_p_raw"] for r in results])
    fdr_q = bh_fdr(pvals)
    for r, q in zip(results, fdr_q):
        r["bh_fdr_q"] = float(q)
        r["sig_05"] = bool(q < 0.05)
    results.sort(key=lambda r: -r["mean_delta_tpr"])
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return {
        "file_type": ft,
        "metric": metric,
        "seeds_used": SEEDS,
        "full_baseline_per_seed": {str(s): full_per_seed[s] for s in SEEDS},
        "groups_ranked": results,
        "top_3": [r["group"] for r in results[:3]],
        "top_5": [r["group"] for r in results[:5]],
        "negative_delta_groups": [r["group"] for r in results if r["mean_delta_tpr"] < 0],
    }


def kendall_tau_vs_global(per_type_ranking: List[str], global_ranking: List[str]) -> float:
    rank_map_pt = {g: i for i, g in enumerate(per_type_ranking)}
    rank_map_gl = {g: i for i, g in enumerate(global_ranking)}
    ranks_pt = np.array([rank_map_pt[g] for g in GROUP_CODES])
    ranks_gl = np.array([rank_map_gl[g] for g in GROUP_CODES])
    tau, _ = stats.kendalltau(ranks_pt, ranks_gl)
    return float(tau)


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-type LOGO results")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--global_ranking_json", default=None,
                    help="Path to inputs_from_p0y/top_k_groups_for_p01.json for kendall tau comparison")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_type = {}
    for ft in PE_FILE_TYPES:
        per_type[ft] = aggregate_one_type(results_dir, ft)

    global_ranking: Optional[List[str]] = None
    kendall: Dict[str, float] = {}
    top3_overlap: Dict[str, float] = {}
    if args.global_ranking_json:
        gl = json.loads(Path(args.global_ranking_json).read_text())
        global_ranking = [r["group"] for r in gl["groups_ranked"]]
        for ft in PE_FILE_TYPES:
            pt_ranking = [r["group"] for r in per_type[ft]["groups_ranked"]]
            kendall[ft] = kendall_tau_vs_global(pt_ranking, global_ranking)
            top3_overlap[ft] = len(set(per_type[ft]["top_3"]) & set(global_ranking[:3])) / 3.0

    out = {
        "metric": PRIMARY_METRIC,
        "n_seeds": len(SEEDS),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_ranking_used": global_ranking,
        "per_type": per_type,
        "global_vs_per_type": {
            "kendall_tau": kendall,
            "top3_overlap_fraction": top3_overlap,
            "interpretation": (
                "tau >= 0.7 → global ranking sufficient; "
                "tau < 0.7 → use per-type top-K in P01 v2 MG-EP-PT."
            ),
        },
    }
    json_path = out_dir / "top_k_per_type.json"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {json_path}")

    md = [
        "# Per-type group LOGO ranking",
        "",
        f"- Generated: `{out['generated_at_utc']}`",
        f"- Metric: `{PRIMARY_METRIC}`",
        f"- Seeds: {SEEDS} (n={out['n_seeds']})",
        "",
    ]
    for ft in PE_FILE_TYPES:
        md.append(f"## {ft}")
        md.append("")
        md.append("| Rank | Group | Mean ΔTPR | 95% CI | Sig |")
        md.append("|---:|:---|---:|:---|:---:|")
        for r in per_type[ft]["groups_ranked"]:
            sig = "Y" if r["sig_05"] else "."
            md.append(f"| {r['rank']} | {r['group']} | {r['mean_delta_tpr']:+.4f} | "
                      f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | {sig} |")
        md.append("")
        md.append(f"**Top-3:** {', '.join(per_type[ft]['top_3'])}")
        md.append(f"**Top-5:** {', '.join(per_type[ft]['top_5'])}")
        md.append(f"**Negative ΔTPR groups:** {', '.join(per_type[ft]['negative_delta_groups']) or 'none'}")
        md.append("")
    if kendall:
        md.append("## Consistency vs global ranking")
        md.append("")
        md.append("| Type | Kendall tau | Top-3 overlap |")
        md.append("|:---|---:|---:|")
        for ft in PE_FILE_TYPES:
            md.append(f"| {ft} | {kendall[ft]:.3f} | {top3_overlap[ft]:.2f} |")
        md.append("")
        md.append(f"_{out['global_vs_per_type']['interpretation']}_")
        md.append("")
    md_path = out_dir / "top_k_per_type.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
