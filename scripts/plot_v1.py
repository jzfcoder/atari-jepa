"""V1 analysis plots.

Reads results/v1/v1_results.json and produces:
  1. Grouped bar chart: mean reward per perturbation level for each condition
  2. Robustness ratio line plot: reward / clean_reward across perturbation levels
  3. Clean vs Robustness Pareto scatter
  4. Per-perturbation degradation heatmap

Usage:
    uv run python scripts/plot_v1.py
    uv run python scripts/plot_v1.py --results results/v1/v1_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Condition display names (matching run_v1.py order)
DISPLAY_NAMES = {
    "stock_cnn": "Stock CNN",
    "jepa_frozen": "JEPA (frozen)",
    "jepa_finetune": "JEPA (fine-tune)",
    "ae_finetune": "AE (fine-tune)",
    "vit_scratch": "ViT (scratch)",
}

CONDITION_ORDER = ["stock_cnn", "jepa_frozen", "jepa_finetune", "ae_finetune", "vit_scratch"]

CONDITION_COLORS = {
    "stock_cnn": "#1f77b4",
    "jepa_frozen": "#2ca02c",
    "jepa_finetune": "#ff7f0e",
    "ae_finetune": "#d62728",
    "vit_scratch": "#9467bd",
}

PERTURBATION_ORDER = ["clean", "color_jitter", "noise", "mild", "hard"]
PERTURBATION_LABELS = {
    "clean": "Clean",
    "color_jitter": "Color Jitter",
    "noise": "Noise",
    "mild": "Mild",
    "hard": "Hard",
}


def plot_grouped_bar(results: dict, output_dir: Path) -> None:
    """Grouped bar chart: mean reward +/- std per perturbation level."""
    conditions = [c for c in CONDITION_ORDER if c in results]
    levels = [l for l in PERTURBATION_ORDER if l in results[conditions[0]]]
    n_levels = len(levels)
    n_conds = len(conditions)

    x = np.arange(n_levels)
    width = 0.8 / n_conds

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, cond in enumerate(conditions):
        means = [results[cond][lv]["mean"] for lv in levels]
        stds = [results[cond][lv]["std"] for lv in levels]
        offset = (i - n_conds / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=3,
            label=DISPLAY_NAMES[cond],
            color=CONDITION_COLORS[cond],
            alpha=0.85,
        )

    ax.set_xlabel("Perturbation Level", fontsize=12)
    ax.set_ylabel("Mean Episodic Return", fontsize=12)
    ax.set_title("V1: Robustness to Visual Perturbations (Breakout)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([PERTURBATION_LABELS.get(l, l) for l in levels])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "v1_robustness_bar.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_robustness_ratio(results: dict, output_dir: Path) -> None:
    """Line plot: reward / clean_reward per perturbation level."""
    conditions = [c for c in CONDITION_ORDER if c in results]
    levels = [l for l in PERTURBATION_ORDER if l in results[conditions[0]]]

    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in conditions:
        clean_mean = results[cond]["clean"]["mean"]
        if clean_mean <= 0:
            continue
        ratios = [results[cond][lv]["mean"] / clean_mean for lv in levels]
        ax.plot(
            range(len(levels)), ratios,
            marker="o", linewidth=2.5, markersize=8,
            label=DISPLAY_NAMES[cond],
            color=CONDITION_COLORS[cond],
        )

    ax.set_xlabel("Perturbation Level", fontsize=12)
    ax.set_ylabel("Reward / Clean Reward", fontsize=12)
    ax.set_title("V1: Robustness Ratio by Perturbation Level", fontsize=14)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([PERTURBATION_LABELS.get(l, l) for l in levels])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.15)
    plt.tight_layout()

    path = output_dir / "v1_robustness_ratio.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pareto(results: dict, output_dir: Path) -> None:
    """Scatter: clean reward vs robustness ratio for each condition."""
    conditions = [c for c in CONDITION_ORDER if c in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    for cond in conditions:
        clean_mean = results[cond]["clean"]["mean"]
        clean_std = results[cond]["clean"]["std"]
        if clean_mean <= 0:
            continue
        perturbed_levels = [l for l in PERTURBATION_ORDER if l != "clean"]
        ratios = [results[cond][lv]["mean"] / clean_mean for lv in perturbed_levels]
        avg_ratio = float(np.mean(ratios))

        ax.scatter(
            clean_mean, avg_ratio,
            s=200, zorder=5,
            color=CONDITION_COLORS[cond],
            edgecolors="black", linewidth=0.8,
        )
        ax.errorbar(
            clean_mean, avg_ratio,
            xerr=clean_std,
            fmt="none", color=CONDITION_COLORS[cond], alpha=0.5, capsize=4,
        )
        ax.annotate(
            DISPLAY_NAMES[cond],
            (clean_mean, avg_ratio),
            textcoords="offset points",
            xytext=(10, 8), fontsize=10,
        )

    ax.set_xlabel("Clean Episodic Return", fontsize=12)
    ax.set_ylabel("Robustness Ratio (avg perturbed / clean)", fontsize=12)
    ax.set_title("V1: Clean Performance vs Robustness", fontsize=14)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()

    path = output_dir / "v1_pareto.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_degradation_heatmap(results: dict, output_dir: Path) -> None:
    """Heatmap: absolute reward drop from clean for each condition x perturbation."""
    conditions = [c for c in CONDITION_ORDER if c in results]
    perturbed_levels = [l for l in PERTURBATION_ORDER if l != "clean"]

    matrix = []
    for cond in conditions:
        clean = results[cond]["clean"]["mean"]
        row = [clean - results[cond][lv]["mean"] for lv in perturbed_levels]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(perturbed_levels)))
    ax.set_xticklabels([PERTURBATION_LABELS.get(l, l) for l in perturbed_levels])
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([DISPLAY_NAMES[c] for c in conditions])

    # Annotate cells
    for i in range(len(conditions)):
        for j in range(len(perturbed_levels)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=11)

    ax.set_title("V1: Reward Drop from Clean (lower = more robust)", fontsize=14)
    fig.colorbar(im, ax=ax, label="Reward Drop")
    plt.tight_layout()

    path = output_dir / "v1_degradation_heatmap.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 analysis plots")
    parser.add_argument(
        "--results", type=str, default="results/v1/v1_results.json",
        help="Path to v1_results.json",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/v1/plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)

    print("Generating V1 analysis plots...")
    plot_grouped_bar(results, output_dir)
    plot_robustness_ratio(results, output_dir)
    plot_pareto(results, output_dir)
    plot_degradation_heatmap(results, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
