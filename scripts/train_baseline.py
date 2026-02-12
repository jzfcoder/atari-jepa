"""Train PPO on Breakout (or another Atari game).

Usage:
    uv run python scripts/train_baseline.py
    uv run python scripts/train_baseline.py --config configs/v0.yaml --seed 123
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agents.ppo_atari import train


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file and return it as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Atari")
    parser.add_argument(
        "--config", type=str, default="configs/v0.yaml",
        help="Path to YAML config file (default: configs/v0.yaml)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the seed from the config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Override seed if provided on the command line.
    if args.seed is not None:
        config["seed"] = args.seed
        print(f"Overriding seed to {args.seed}")

    print("=" * 60)
    print("Training configuration:")
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")
    print("=" * 60)

    model_path = train(config)

    print("=" * 60)
    print(f"Training complete. Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
