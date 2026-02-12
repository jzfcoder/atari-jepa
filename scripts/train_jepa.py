"""Train the JEPA self-supervised encoder on collected Atari frames.

Usage:
    uv run python scripts/train_jepa.py
    uv run python scripts/train_jepa.py --config configs/jepa.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agents.jepa import train_jepa


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file and return it as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JEPA encoder on Atari frames")
    parser.add_argument(
        "--config", type=str, default="configs/jepa.yaml",
        help="Path to YAML config file (default: configs/jepa.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print("JEPA training configuration:")
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")
    print("=" * 60)

    encoder_path = train_jepa(config)

    print("=" * 60)
    print(f"JEPA training complete. Encoder saved to: {encoder_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
