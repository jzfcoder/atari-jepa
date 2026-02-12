"""Evaluate a trained agent under visual perturbations and produce a comparison table.

Usage:
    uv run python scripts/measure_gap.py --model results/v0/.../final_model.pt
    uv run python scripts/measure_gap.py --model results/v0/.../final_model.pt --num-episodes 50 --seeds 42,123,456
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agents.ppo_atari import evaluate
from env.perturbations import make_perturbed_atari_env


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def evaluate_condition(
    model_path: str,
    env_id: str,
    perturbation_level: str,
    num_episodes: int,
    seeds: list[int],
    device: str,
) -> dict:
    """Evaluate a model across multiple seeds and aggregate the results."""
    all_rewards: list[float] = []

    for seed in seeds:
        def env_fn(_lvl=perturbation_level, _seed=seed):
            return make_perturbed_atari_env(
                env_id, perturbation_level=_lvl, seed=_seed, training=False,
            )

        results = evaluate(
            model_path=model_path,
            env_fn=env_fn,
            num_episodes=num_episodes,
            device=device,
        )
        all_rewards.extend(results["rewards"])

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "all_rewards": all_rewards,
    }


def format_table(rows: list[tuple[str, float, float, float]]) -> str:
    """Format results into a readable table."""
    header = f"{'Condition':<22}| {'Mean Reward':>11} | {'Std':>6} | {'Robustness Ratio':>16}"
    separator = "-" * len(header)
    lines = [separator, header, separator]

    for name, mean_r, std_r, ratio in rows:
        lines.append(
            f"{name:<22}| {mean_r:>11.1f} | {std_r:>6.1f} | {ratio:>16.3f}"
        )

    lines.append(separator)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure visual robustness gap for a trained agent"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the saved model checkpoint",
    )
    parser.add_argument(
        "--env-id", type=str, default="ALE/Breakout-v5",
        help="Gymnasium environment ID (default: ALE/Breakout-v5)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=50,
        help="Number of episodes per seed per condition (default: 50)",
    )
    parser.add_argument(
        "--seeds", type=str, default="42,123,456",
        help="Comma-separated list of evaluation seeds (default: 42,123,456)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for inference (default: cpu)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/v0",
        help="Directory to save the results table (default: results/v0)",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # (display_name, perturbation_level)
    conditions = [
        ("Clean", "clean"),
        ("Color Jitter", "color_jitter"),
        ("Gaussian Noise", "noise"),
        ("Both (mild)", "mild"),
        ("Both (hard)", "hard"),
    ]

    # ------------------------------------------------------------------
    # Run evaluations
    # ------------------------------------------------------------------
    results_by_condition: list[tuple[str, float, float]] = []

    for name, level in conditions:
        print(f"Evaluating: {name} ({len(seeds)} seeds x {args.num_episodes} episodes)...")
        res = evaluate_condition(
            model_path=args.model,
            env_id=args.env_id,
            perturbation_level=level,
            num_episodes=args.num_episodes,
            seeds=seeds,
            device=args.device,
        )
        results_by_condition.append((name, res["mean_reward"], res["std_reward"]))
        print(f"  -> Mean: {res['mean_reward']:.1f}, Std: {res['std_reward']:.1f}")

    # ------------------------------------------------------------------
    # Compute robustness ratios and format the table
    # ------------------------------------------------------------------
    clean_mean = results_by_condition[0][1]
    table_rows: list[tuple[str, float, float, float]] = []
    for name, mean_r, std_r in results_by_condition:
        ratio = mean_r / clean_mean if clean_mean > 0 else 0.0
        table_rows.append((name, mean_r, std_r, ratio))

    table_str = format_table(table_rows)

    print("\n" + table_str)

    # ------------------------------------------------------------------
    # Save to file
    # ------------------------------------------------------------------
    results_path = output_dir / "perturbation_results.txt"
    with open(results_path, "w") as f:
        f.write("Visual Perturbation Robustness Results\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Env: {args.env_id}\n")
        f.write(f"Episodes per seed: {args.num_episodes}\n")
        f.write(f"Seeds: {seeds}\n\n")
        f.write(table_str)
        f.write("\n")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
