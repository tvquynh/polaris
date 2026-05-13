"""aggregate_phase4.py — Master aggregation: Phase 4a (no HPO) + Phase 4b OPTUNA100.

For each config × 10 seeds:
    - Mean ± std TPR (ensemble + per-type)
    - 95% bootstrap CI
    - Paired Wilcoxon signed-rank vs baseline_B (n=10 paired)
    - Cohen's d effect size
    - BH-FDR adjusted q across configs

Output:
    aggregated_master_table.json   (machine-readable, source of truth)
    aggregated_master_table.md     (human-readable, manuscript table)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

# Configs to include in Phase 4 master table (n=10 required)
# Skip: baseline_A (per memory rule), drop_bottom_* (failed bug), fte_ls_hpo (only n=2)
CONFIGS_ORDERED = [
    "baseline_B",
    "fte",
    "fte_sar",
    "fte_mgep_top3",
    "fte_mgep_top5",
    "fte_mgep_sar",
    "fte_beh_only",
    "fte_ls",
    "fte_mgep_sar_ls",
    "fte_mgep_sar_ls_peto_noHPO",
    "R_5",
    "R_20",
    "fte_ls_optuna100",
]

METRIC_KEYS = {
    "ensemble_tpr_001": ("ensemble", "ensemble_tpr_at_fpr_0.01"),
    "win32_tpr_001": ("per_type", "win32", "tpr_at_fpr_0.01_challenge"),
    "win64_tpr_001": ("per_type", "win64", "tpr_at_fpr_0.01_challenge"),
    "dot_net_tpr_001": ("per_type", "dot_net", "tpr_at_fpr_0.01_challenge"),
}


def get_nested(d, keys):
    for k in keys:
        d = d[k]
    return d


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                  rng=None) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def cohen_d_paired(diffs: np.ndarray) -> float:
    """Cohen's d for paired differences."""
    if len(diffs) < 2:
        return float("nan")
    mean_d = np.mean(diffs)
    sd_d = np.std(diffs, ddof=1)
    if sd_d < 1e-12:
        return float("inf") if abs(mean_d) > 0 else 0.0
    return float(mean_d / sd_d)


def load_config_metrics(results_dir: Path, cfg: str) -> dict:
    """Returns dict {metric_key: np.array(values_per_seed)} for one config."""
    out = {k: [] for k in METRIC_KEYS}
    out["seeds"] = []
    for seed_dir in sorted(results_dir.glob(f"{cfg}/seed_*")):
        mp = seed_dir / "metrics.json"
        if not mp.exists():
            continue
        d = json.loads(mp.read_text())
        out["seeds"].append(int(seed_dir.name.split("_")[1]))
        for k, path in METRIC_KEYS.items():
            out[k].append(float(get_nested(d, path)))
    for k in METRIC_KEYS:
        out[k] = np.array(out[k])
    return out


def aggregate(results_dir: Path) -> dict:
    rng = np.random.default_rng(42)
    baseline_data = load_config_metrics(results_dir, "baseline_B")
    if len(baseline_data["seeds"]) == 0:
        raise RuntimeError("baseline_B has no seed results — cannot compute Δ")

    configs_out: list[dict] = []
    raw_pvals_per_metric: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}

    for cfg in CONFIGS_ORDERED:
        cdata = load_config_metrics(results_dir, cfg)
        n = len(cdata["seeds"])
        entry = {"config": cfg, "n_seeds": n, "seeds_used": cdata["seeds"], "metrics": {}}

        for mkey in METRIC_KEYS:
            vals = cdata[mkey]
            if len(vals) == 0:
                entry["metrics"][mkey] = None
                raw_pvals_per_metric[mkey].append(1.0)
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_lo, ci_hi = bootstrap_ci(vals, rng=rng) if len(vals) > 1 else (mean, mean)

            # Paired stats vs baseline_B (only if same seeds, n=10)
            bvals = baseline_data[mkey]
            paired_diff_mean = None
            wilcoxon_p = None
            cohen_d = None
            if cfg != "baseline_B" and len(vals) == len(bvals):
                # Align by seed
                base_seed_map = {s: v for s, v in zip(baseline_data["seeds"], bvals)}
                aligned_base = np.array([base_seed_map[s] for s in cdata["seeds"]])
                diffs = vals - aligned_base
                paired_diff_mean = float(np.mean(diffs))
                try:
                    _, wilcoxon_p = stats.wilcoxon(diffs, alternative="two-sided",
                                                     zero_method="wilcox")
                    wilcoxon_p = float(wilcoxon_p)
                except ValueError:
                    wilcoxon_p = 1.0
                cohen_d = cohen_d_paired(diffs)

            entry["metrics"][mkey] = {
                "mean": mean, "std": std,
                "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                "min": float(np.min(vals)), "max": float(np.max(vals)),
                "values": [float(v) for v in vals],
                "paired_diff_mean_vs_baseline_B": paired_diff_mean,
                "wilcoxon_p_raw": wilcoxon_p,
                "cohen_d": cohen_d,
            }
            raw_pvals_per_metric[mkey].append(wilcoxon_p if wilcoxon_p is not None else 1.0)

        configs_out.append(entry)

    # BH-FDR adjust per-metric across configs
    for mkey in METRIC_KEYS:
        pvals = np.array(raw_pvals_per_metric[mkey])
        qvals = bh_fdr(pvals)
        for i, cfg_entry in enumerate(configs_out):
            if cfg_entry["metrics"].get(mkey):
                cfg_entry["metrics"][mkey]["bh_fdr_q"] = float(qvals[i])
                cfg_entry["metrics"][mkey]["fdr_sig_05"] = bool(qvals[i] < 0.05)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "primary_metric": "ensemble_tpr_001",
        "n_configs": len(CONFIGS_ORDERED),
        "fdr_method": "BH q<0.05",
        "wilcoxon_method": "two-sided signed-rank, paired by seed",
        "configs": configs_out,
    }


def render_md(agg: dict) -> str:
    L = []
    L.append("# P01 v2 Master Aggregation — Phase 4a + Phase 4b OPTUNA100\n")
    L.append(f"Generated: `{agg['generated_at_utc']}`\n")
    L.append(f"FDR: {agg['fdr_method']} · Wilcoxon: {agg['wilcoxon_method']}\n")

    # Ensemble TPR table (primary)
    L.append("\n## Primary: TPR @ 1%FPR challenge (ENSEMBLE)\n")
    L.append("| Rank | Config | n | Mean | Std | 95% CI | Δ vs baseline_B | Wilcoxon p | BH-FDR q | Cohen's d | Sig |")
    L.append("|:--:|:---|:--:|---:|---:|:---|---:|---:|---:|---:|:--:|")
    ranked = sorted(agg["configs"],
                    key=lambda c: -(c["metrics"].get("ensemble_tpr_001") or {}).get("mean", -1))
    for i, c in enumerate(ranked, 1):
        m = c["metrics"].get("ensemble_tpr_001")
        if not m:
            continue
        sig = "Y" if m.get("fdr_sig_05") else "."
        delta = m.get("paired_diff_mean_vs_baseline_B")
        wp = m.get("wilcoxon_p_raw")
        cd = m.get("cohen_d")
        delta_str = f"{delta*100:+.2f} pp" if delta is not None else "—"
        wp_str = f"{wp:.4g}" if wp is not None else "—"
        cd_str = f"{cd:.2f}" if cd is not None and abs(cd) < 1000 else "—"
        qv = m.get("bh_fdr_q", 1.0)
        L.append(f"| {i} | `{c['config']}` | {c['n_seeds']} | {m['mean']:.4f} | {m['std']:.4f} | "
                 f"[{m['ci95_lo']:.4f}, {m['ci95_hi']:.4f}] | {delta_str} | {wp_str} | {qv:.4f} | {cd_str} | {sig} |")

    # Per-type breakdown for top 5 configs
    L.append("\n## Per-type breakdown — top 5 configs by ensemble TPR\n")
    L.append("| Config | Win32 (n) | Win64 (n) | .NET (n) |")
    L.append("|:---|---:|---:|---:|")
    for c in ranked[:5]:
        cells = []
        for ft in ["win32", "win64", "dot_net"]:
            mk = f"{ft}_tpr_001"
            m = c["metrics"].get(mk) or {}
            if "mean" in m:
                cells.append(f"{m['mean']:.4f} ± {m['std']:.4f}")
            else:
                cells.append("—")
        L.append(f"| `{c['config']}` | {cells[0]} | {cells[1]} | {cells[2]} |")

    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = aggregate(results_dir)
    (out_dir / "aggregated_master_table.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "aggregated_master_table.md").write_text(render_md(agg), encoding="utf-8")

    print(f"Wrote: {out_dir / 'aggregated_master_table.json'}")
    print(f"Wrote: {out_dir / 'aggregated_master_table.md'}")

    print("\nTop-5 ENSEMBLE TPR ranking:")
    ranked = sorted(agg["configs"],
                    key=lambda c: -(c["metrics"].get("ensemble_tpr_001") or {}).get("mean", -1))
    for i, c in enumerate(ranked[:5], 1):
        m = c["metrics"]["ensemble_tpr_001"]
        delta = m.get("paired_diff_mean_vs_baseline_B")
        delta_str = f"{delta*100:+.2f} pp" if delta else "ref"
        print(f"  #{i} {c['config']:<35} mean={m['mean']:.4f} std={m['std']:.4f} Δ={delta_str}")
