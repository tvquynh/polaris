"""generate_figures.py — Phase 4 figures for P01 v2 manuscript.

Per memory `feedback_paper_style_minimalist.md`: monochrome, white box + thin
black border + grey banners, NO bright colors.

Figures:
    fig1_ablation_bar      — main TPR bar chart (12 configs) with 95% CI error
    fig2_per_type_breakdown — grouped bars Win32/Win64/.NET top 5 configs
    fig3_roc_curves        — ROC curves baseline vs fte_ls_optuna100
    fig4_compute_vs_tpr    — scatter compute cost vs TPR (Pareto front)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_master(results_dir: Path) -> dict:
    return json.loads((results_dir / "aggregated" / "aggregated_master_table.json").read_text())


HUMAN_LABEL = {
    "baseline_B": "Baseline (Joyce-style)",
    "baseline_B_optuna": "Baseline + Optuna",
    "fte": "FTE",
    "fte_sar": "FTE + SAR",
    "fte_mgep_top3": "FTE + MG-EP-PT",
    "fte_mgep_sar": "FTE + MG + SAR",
    "fte_beh_only": "FTE + BEH-EP (legacy)",
    "fte_ls": "FTE + LS",
    "fte_ls_mgep": "FTE + LS + MG-EP-PT",
    "fte_ls_sar": "FTE + LS + SAR",
    "fte_ls_peto": "FTE + LS + PETO",
    "fte_mgep_sar_ls": "FTE + MG + SAR + LS",
    "fte_mgep_sar_ls_peto_noHPO": "FTE + 4-component",
    "R_5": "4-component, R=5",
    "R_20": "4-component, R=20",
    "fte_ls_optuna100": "FTE + LS + Optuna",
    "fte_ls_peto_optuna100": "FTE + LS + PETO + Optuna",
}


def fig1_ablation_bar(master: dict, out_path: Path):
    """Main TPR bar chart, configs ranked descending. Highlight NEW winner."""
    configs = master["configs"]
    ranked = sorted(configs,
                    key=lambda c: -(c["metrics"].get("ensemble_tpr_001") or {}).get("mean", -1))

    names = [c["config"] for c in ranked]
    labels = [HUMAN_LABEL.get(n, n) for n in names]
    means = [c["metrics"]["ensemble_tpr_001"]["mean"] for c in ranked]
    ci_los = [c["metrics"]["ensemble_tpr_001"]["mean"] - c["metrics"]["ensemble_tpr_001"]["ci95_lo"]
              for c in ranked]
    ci_his = [c["metrics"]["ensemble_tpr_001"]["ci95_hi"] - c["metrics"]["ensemble_tpr_001"]["mean"]
              for c in ranked]
    errs = np.array([ci_los, ci_his])

    # Highlight: new winner = darkest, prior winner = dark, baseline = lightest
    colors = []
    for name in names:
        if name == "fte_ls_peto_optuna100":
            colors.append("#111111")  # NEW winner = black
        elif name == "fte_ls_optuna100":
            colors.append("#444444")  # Prior winner = dark grey
        elif name == "baseline_B":
            colors.append("#cccccc")  # baseline = lightest
        else:
            colors.append("#888888")  # others = mid grey

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=errs, color=colors, edgecolor="black", linewidth=0.5,
           capsize=2, error_kw={"linewidth": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("TPR @ 1% FPR (challenge set)")
    ax.set_ylim(0.55, 0.80)
    # Baseline TPR reference line
    baseline_mean = next((c["metrics"]["ensemble_tpr_001"]["mean"] for c in ranked
                          if c["config"] == "baseline_B"), None)
    if baseline_mean is not None:
        ax.axhline(baseline_mean, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linewidth=0.3, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def fig2_per_type_breakdown(master: dict, out_path: Path):
    """Grouped bars: Win32 / Win64 / .NET for top 5 configs."""
    top5 = sorted(master["configs"],
                  key=lambda c: -(c["metrics"].get("ensemble_tpr_001") or {}).get("mean", -1))[:5]
    types = ["win32", "win64", "dot_net"]
    type_labels = ["Win32", "Win64", ".NET"]

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.15
    x = np.arange(len(types))
    grey_shades = ["#222222", "#444444", "#777777", "#aaaaaa", "#cccccc"]
    for i, c in enumerate(top5):
        means = [c["metrics"][f"{t}_tpr_001"]["mean"] for t in types]
        stds = [c["metrics"][f"{t}_tpr_001"]["std"] for t in types]
        ax.bar(x + i * width, means, width, yerr=stds, label=c["config"],
               color=grey_shades[i], edgecolor="black", linewidth=0.4,
               capsize=1.5, error_kw={"linewidth": 0.5})
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(type_labels)
    ax.set_ylabel("TPR @ 1% FPR")
    ax.set_ylim(0.4, 0.95)
    ax.legend(fontsize=7, frameon=True, edgecolor="black", loc="lower left")
    ax.grid(axis="y", linewidth=0.3, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def fig3_roc_curves(results_dir: Path, out_path: Path):
    """ROC: baseline_B vs fte_ls_optuna100 vs fte_ls (seed 42, ensemble)."""
    from sklearn.metrics import roc_curve

    cfgs_plot = [
        ("baseline_B", "Baseline (Joyce KDD'25)", "#bbbbbb", "-"),
        ("fte_ls", "FTE + LS (no HPO)", "#666666", "--"),
        ("fte_ls_optuna100", "FTE + LS + Optuna params", "#222222", "-"),
    ]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for cfg, label, color, ls in cfgs_plot:
        seed_dir = results_dir / cfg / "seed_42"
        if not seed_dir.exists():
            continue
        sb_all, sm_all = [], []
        for ft in ["win32", "win64", "dot_net"]:
            b_npz = seed_dir / f"predictions_test_benign_{ft}.npz"
            m_npz = seed_dir / f"predictions_challenge_{ft}.npz"
            if not b_npz.exists() or not m_npz.exists():
                continue
            sb_all.extend(np.load(b_npz)["scores"].tolist())
            sm_all.extend(np.load(m_npz)["scores"].tolist())
        if not sb_all:
            continue
        y = np.concatenate([np.zeros(len(sb_all)), np.ones(len(sm_all))])
        s = np.concatenate([sb_all, sm_all])
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.2, label=label)

    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0, 1)
    ax.axvline(0.01, color="black", linewidth=0.4, linestyle=":", alpha=0.5)
    ax.text(0.011, 0.05, "FPR=1%", fontsize=8, alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8, frameon=True, edgecolor="black")
    ax.grid(linewidth=0.3, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def fig4_compute_vs_tpr(master: dict, results_dir: Path, out_path: Path):
    """Scatter: compute cost (min/seed) vs TPR. Pareto front."""
    import glob
    configs_to_show = ["baseline_B", "fte", "fte_ls", "fte_mgep_sar_ls",
                       "fte_mgep_sar_ls_peto_noHPO", "fte_ls_optuna100", "fte_ls_hpo"]
    points = []
    for c in master["configs"]:
        if c["config"] not in configs_to_show:
            continue
        cfg = c["config"]
        # Get wallclock from metrics.json files
        walls = []
        for p in sorted((results_dir / cfg).glob("seed_*/metrics.json")):
            d = json.loads(p.read_text())
            walls.append(d.get("wallclock_s", 0) / 60)
        if not walls:
            continue
        mean_wall = sum(walls) / len(walls)
        m = c["metrics"].get("ensemble_tpr_001")
        if not m:
            continue
        points.append((mean_wall, m["mean"], cfg))

    # Add fte_ls_hpo separately (n=2 only, parsed manually)
    hpo_walls = []
    for p in sorted((results_dir / "fte_ls_hpo").glob("seed_*/metrics.json")):
        d = json.loads(p.read_text())
        hpo_walls.append(d.get("wallclock_s", 0) / 60)
        hpo_tpr = d["ensemble"]["ensemble_tpr_at_fpr_0.01"]
    if hpo_walls:
        mean_wall = sum(hpo_walls) / len(hpo_walls)
        # Use mean TPR from existing files
        tprs = [json.loads(p.read_text())["ensemble"]["ensemble_tpr_at_fpr_0.01"]
                for p in sorted((results_dir / "fte_ls_hpo").glob("seed_*/metrics.json"))]
        if tprs:
            points.append((mean_wall, sum(tprs) / len(tprs), "fte_ls_hpo (in-loop HPO)"))

    fig, ax = plt.subplots(figsize=(7, 4))
    for wall, tpr, name in points:
        marker_size = 80 if name == "fte_ls_optuna100" else 40
        face = "#222222" if name == "fte_ls_optuna100" else "white"
        ax.scatter(wall, tpr, s=marker_size, c=face, edgecolors="black",
                   linewidths=0.8, zorder=3)
        ax.annotate(name, (wall, tpr), xytext=(5, 5), textcoords="offset points",
                    fontsize=7, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("Compute cost (min/seed, log scale)")
    ax.set_ylabel("TPR @ 1% FPR (challenge)")
    ax.grid(linewidth=0.3, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master(results_dir)

    print("Generating fig1_ablation_bar.pdf ...")
    fig1_ablation_bar(master, out_dir / "fig1_ablation_bar.pdf")

    print("Generating fig2_per_type_breakdown.pdf ...")
    fig2_per_type_breakdown(master, out_dir / "fig2_per_type_breakdown.pdf")

    print("Generating fig3_roc_curves.pdf ...")
    fig3_roc_curves(results_dir, out_dir / "fig3_roc_curves.pdf")

    print("Generating fig4_compute_vs_tpr.pdf ...")
    fig4_compute_vs_tpr(master, results_dir, out_dir / "fig4_compute_vs_tpr.pdf")

    print(f"\nAll figures saved to: {out_dir}")
    for f in sorted(out_dir.glob("fig*.pdf")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
