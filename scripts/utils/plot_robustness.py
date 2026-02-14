"""Plot Phase 3 robustness results.

Reads phase3_results.json and produces:
  1. Grouped bar chart: mean reward per perturbation level for each encoder
  2. Robustness ratio line plot: reward / clean_reward per perturbation level

Usage:
    uv run python scripts/plot_robustness.py
    uv run python scripts/plot_robustness.py --results results/v0/phase3/phase3_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITION_COLORS = {
    "Stock CNN": "#1f77b4",
    "JEPA": "#2ca02c",
    "Autoencoder": "#d62728",
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
    conditions = list(results.keys())
    levels = [l for l in PERTURBATION_ORDER if l in next(iter(results.values()))]
    n_levels = len(levels)
    n_conds = len(conditions)

    x = np.arange(n_levels)
    width = 0.8 / n_conds

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cond in enumerate(conditions):
        means = [results[cond][lv]["mean"] for lv in levels]
        stds = [results[cond][lv]["std"] for lv in levels]
        offset = (i - n_conds / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=3,
            label=cond,
            color=CONDITION_COLORS.get(cond, None),
            alpha=0.85,
        )

    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("Mean Episodic Return")
    ax.set_title("Robustness to Visual Perturbations (Breakout)")
    ax.set_xticks(x)
    ax.set_xticklabels([PERTURBATION_LABELS.get(l, l) for l in levels])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "robustness_bar.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_robustness_ratio(results: dict, output_dir: Path) -> None:
    """Line plot: reward / clean_reward per perturbation level."""
    conditions = list(results.keys())
    levels = [l for l in PERTURBATION_ORDER if l in next(iter(results.values()))]

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in conditions:
        clean_mean = results[cond]["clean"]["mean"]
        if clean_mean <= 0:
            continue
        ratios = [results[cond][lv]["mean"] / clean_mean for lv in levels]
        ax.plot(
            range(len(levels)), ratios,
            marker="o", linewidth=2, markersize=7,
            label=cond,
            color=CONDITION_COLORS.get(cond, None),
        )

    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("Reward / Clean Reward")
    ax.set_title("Robustness Ratio by Perturbation Level")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([PERTURBATION_LABELS.get(l, l) for l in levels])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (1.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.3)
    plt.tight_layout()

    path = output_dir / "robustness_ratio.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase 3 robustness results")
    parser.add_argument(
        "--results", type=str, default="results/v0/phase3/phase3_results.json",
        help="Path to phase3_results.json",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/v0/phase4",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)

    print("Plotting robustness results...")
    plot_grouped_bar(results, output_dir)
    plot_robustness_ratio(results, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
