"""Train the autoencoder baseline encoder on collected Atari frames.

Usage:
    uv run python scripts/train_autoencoder.py
    uv run python scripts/train_autoencoder.py --config configs/autoencoder.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agents.autoencoder import train_autoencoder


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file and return it as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train autoencoder encoder on Atari frames"
    )
    parser.add_argument(
        "--config", type=str, default="configs/autoencoder.yaml",
        help="Path to YAML config file (default: configs/autoencoder.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print("Autoencoder training configuration:")
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")
    print("=" * 60)

    encoder_path = train_autoencoder(config)

    print("=" * 60)
    print(f"Autoencoder training complete. Encoder saved to: {encoder_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
