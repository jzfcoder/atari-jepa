"""Plot training learning curves from TensorBoard logs.

Extracts charts/episodic_return from TensorBoard event files for each
condition (Stock CNN single seed, JEPA 3 seeds, AE 3 seeds) and plots
smoothed learning curves with mean +/- std shading for multi-seed runs.

Usage:
    uv run python scripts/plot_learning_curves.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CONDITION_COLORS = {
    "Stock CNN": "#1f77b4",
    "JEPA": "#2ca02c",
    "Autoencoder": "#d62728",
}


def extract_scalar(log_dir: str, tag: str = "charts/episodic_return") -> tuple[np.ndarray, np.ndarray]:
    """Extract a scalar from a TensorBoard log directory.

    Returns (steps, values) as numpy arrays.
    """
    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        raise ValueError(f"Tag '{tag}' not found in {log_dir}. Available: {ea.Tags().get('scalars', [])}")
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    # Pad to preserve length
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def interpolate_to_common_steps(
    all_steps: list[np.ndarray],
    all_values: list[np.ndarray],
    num_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate multiple runs to a common step grid.

    Returns (common_steps, values_matrix) where values_matrix is (n_runs, num_points).
    """
    # Find the step range common to all runs
    min_step = max(s[0] for s in all_steps)
    max_step = min(s[-1] for s in all_steps)
    common_steps = np.linspace(min_step, max_step, num_points)

    values_matrix = np.zeros((len(all_steps), num_points))
    for i, (steps, values) in enumerate(zip(all_steps, all_values)):
        values_matrix[i] = np.interp(common_steps, steps, values)

    return common_steps, values_matrix


def plot_learning_curves(
    condition_logs: dict[str, list[str]],
    output_dir: Path,
    smooth_window: int = 50,
) -> None:
    """Plot learning curves for all conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond_name, log_dirs in condition_logs.items():
        color = CONDITION_COLORS.get(cond_name, None)

        all_steps = []
        all_values = []
        for log_dir in log_dirs:
            try:
                steps, values = extract_scalar(log_dir)
                all_steps.append(steps)
                all_values.append(values)
            except (ValueError, Exception) as e:
                print(f"  Warning: skipping {log_dir}: {e}")

        if not all_steps:
            print(f"  No data for {cond_name}, skipping.")
            continue

        if len(all_steps) == 1:
            # Single seed: just plot smoothed
            steps = all_steps[0]
            values = smooth(all_values[0], smooth_window)
            ax.plot(steps, values, label=cond_name, color=color, linewidth=2)
        else:
            # Multiple seeds: interpolate, smooth each, plot mean +/- std
            common_steps, values_matrix = interpolate_to_common_steps(all_steps, all_values)
            smoothed = np.array([smooth(v, smooth_window) for v in values_matrix])
            mean_vals = smoothed.mean(axis=0)
            std_vals = smoothed.std(axis=0)

            ax.plot(common_steps, mean_vals, label=cond_name, color=color, linewidth=2)
            ax.fill_between(
                common_steps,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.2, color=color,
            )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episodic Return")
    ax.set_title("Training Learning Curves (Breakout)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = output_dir / "learning_curves.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def discover_log_dirs(base_dir: str) -> dict[str, list[str]]:
    """Auto-discover TensorBoard log directories from the results tree."""
    base = Path(base_dir)
    logs: dict[str, list[str]] = {}

    # Stock CNN: single seed baseline
    # Look for ppo_Breakout* directories with tb/ subfolder
    stock_dirs = []
    for d in sorted(base.glob("ppo_Breakout-v5_*/tb")):
        stock_dirs.append(str(d))
    if stock_dirs:
        # Use the last one (most recent baseline)
        logs["Stock CNN"] = [stock_dirs[-1]]

    # JEPA: phase3/jepa/jepa_seed*/tb
    jepa_dirs = sorted(str(d) for d in base.glob("phase3/jepa/jepa_seed*/tb"))
    if jepa_dirs:
        logs["JEPA"] = jepa_dirs

    # Autoencoder: phase3/autoencoder/autoencoder_seed*/tb
    ae_dirs = sorted(str(d) for d in base.glob("phase3/autoencoder/autoencoder_seed*/tb"))
    if ae_dirs:
        logs["Autoencoder"] = ae_dirs

    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training learning curves")
    parser.add_argument(
        "--base-dir", type=str, default="results/v0",
        help="Base results directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/v0/phase4",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--smooth", type=int, default=50,
        help="Rolling average window size for smoothing",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering TensorBoard logs...")
    condition_logs = discover_log_dirs(args.base_dir)
    for cond, dirs in condition_logs.items():
        print(f"  {cond}: {len(dirs)} run(s)")
        for d in dirs:
            print(f"    {d}")

    print("\nPlotting learning curves...")
    plot_learning_curves(condition_logs, output_dir, smooth_window=args.smooth)
    print("Done.")


if __name__ == "__main__":
    main()
