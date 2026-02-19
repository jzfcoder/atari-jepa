"""Plot V1 learning curves from TensorBoard logs.

Reads charts/episodic_return from each run's TensorBoard events,
smooths with a rolling window, and plots all conditions.

Usage:
    uv run python scripts/plot_learning_curves_v1.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CONDITION_ORDER = ["stock_cnn", "jepa_frozen", "jepa_finetune", "ae_finetune", "vit_scratch"]

DISPLAY_NAMES = {
    "stock_cnn": "Stock CNN",
    "jepa_frozen": "JEPA (frozen)",
    "jepa_finetune": "JEPA (fine-tune)",
    "ae_finetune": "AE (fine-tune)",
    "vit_scratch": "ViT (scratch)",
}

CONDITION_COLORS = {
    "stock_cnn": "#1f77b4",
    "jepa_frozen": "#2ca02c",
    "jepa_finetune": "#ff7f0e",
    "ae_finetune": "#d62728",
    "vit_scratch": "#9467bd",
}

SEEDS = [1, 2, 3]
RESULTS_DIR = Path("results/v1")
OUTPUT_DIR = Path("results/v1/plots")


def parse_tb_log(tb_dir: Path) -> tuple[list[int], list[float]]:
    """Extract (steps, returns) from TensorBoard event files."""
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if "charts/episodic_return" not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars("charts/episodic_return")
    steps = [e.step for e in events]
    returns = [e.value for e in events]
    return steps, returns


def smooth(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Simple rolling mean."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_per_condition(all_data: dict, output_dir: Path) -> None:
    """One subplot per condition, showing individual seeds + mean."""
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=True)

    for ax, cond in zip(axes, CONDITION_ORDER):
        if cond not in all_data:
            continue
        color = CONDITION_COLORS[cond]

        for seed, (steps, returns) in all_data[cond].items():
            s = np.array(steps)
            r = smooth(np.array(returns), window=50)
            s_smooth = s[:len(r)]
            ax.plot(s_smooth, r, alpha=0.3, color=color, linewidth=0.8)

        # Mean across seeds (interpolate to common x-axis)
        common_x = np.linspace(0, 10_000_000, 500)
        interp_curves = []
        for seed, (steps, returns) in all_data[cond].items():
            r = smooth(np.array(returns), window=50)
            s = np.array(steps[:len(r)])
            if len(s) > 1:
                interp_y = np.interp(common_x, s, r)
                interp_curves.append(interp_y)

        if interp_curves:
            mean_curve = np.mean(interp_curves, axis=0)
            ax.plot(common_x, mean_curve, color=color, linewidth=2.5, label="Mean")

        ax.set_title(DISPLAY_NAMES[cond], fontsize=12)
        ax.set_xlabel("Steps")
        ax.set_xlim(0, 10_000_000)
        ax.grid(alpha=0.3)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))

    axes[0].set_ylabel("Episodic Return", fontsize=12)
    fig.suptitle("V1: Learning Curves by Condition (3 seeds each, smoothed)", fontsize=14)
    plt.tight_layout()

    path = output_dir / "v1_learning_curves_grid.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_combined(all_data: dict, output_dir: Path) -> None:
    """All conditions on one plot (mean + std band)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    common_x = np.linspace(0, 10_000_000, 500)

    for cond in CONDITION_ORDER:
        if cond not in all_data:
            continue
        color = CONDITION_COLORS[cond]

        interp_curves = []
        for seed, (steps, returns) in all_data[cond].items():
            r = smooth(np.array(returns), window=50)
            s = np.array(steps[:len(r)])
            if len(s) > 1:
                interp_y = np.interp(common_x, s, r)
                interp_curves.append(interp_y)

        if interp_curves:
            curves = np.array(interp_curves)
            mean_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            ax.plot(common_x, mean_curve, color=color, linewidth=2.5,
                    label=DISPLAY_NAMES[cond])
            ax.fill_between(common_x, mean_curve - std_curve, mean_curve + std_curve,
                            color=color, alpha=0.15)

    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Episodic Return", fontsize=12)
    ax.set_title("V1: Learning Curves (mean +/- std across 3 seeds)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 10_000_000)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))
    plt.tight_layout()

    path = output_dir / "v1_learning_curves.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse all TensorBoard logs
    all_data: dict[str, dict[int, tuple]] = {}
    for cond in CONDITION_ORDER:
        all_data[cond] = {}
        for seed in SEEDS:
            tb_dir = RESULTS_DIR / cond / f"{cond}_seed{seed}" / "tb"
            if tb_dir.exists():
                steps, returns = parse_tb_log(tb_dir)
                if steps:
                    all_data[cond][seed] = (steps, returns)
                    print(f"  Parsed {cond} seed={seed}: {len(steps)} episodes, "
                          f"final step={steps[-1]:,}")
                else:
                    print(f"  Warning: no data in {tb_dir}")
            else:
                print(f"  Missing: {tb_dir}")

    print("\nGenerating learning curve plots...")
    plot_per_condition(all_data, OUTPUT_DIR)
    plot_combined(all_data, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
