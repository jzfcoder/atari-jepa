"""Train PPO with a frozen pre-trained encoder (JEPA or autoencoder).

The encoder weights are frozen — only the policy and value heads are trained.
This isolates the effect of the encoder's representations on RL performance.

Usage:
    uv run python scripts/train_with_encoder.py --encoder-type jepa \
        --encoder-path results/v0/jepa/encoder_final.pt

    uv run python scripts/train_with_encoder.py --encoder-type autoencoder \
        --encoder-path results/v0/autoencoder/encoder_final.pt \
        --seed 2 --save-dir results/v0/phase3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agents.jepa import load_jepa_encoder
from agents.autoencoder import load_ae_encoder
from agents.ppo_atari import train, get_device, DEFAULT_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO with a frozen pre-trained encoder"
    )
    parser.add_argument(
        "--encoder-type", type=str, required=True,
        choices=["jepa", "autoencoder"],
        help="Type of pre-trained encoder to use",
    )
    parser.add_argument(
        "--encoder-path", type=str, required=True,
        help="Path to the encoder checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional YAML config file for PPO hyperparameters",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--env-id", type=str, default=None)
    args = parser.parse_args()

    # Build config
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.save_dir is not None:
        config["save_dir"] = args.save_dir
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    if args.env_id is not None:
        config["env_id"] = args.env_id

    # Tag the run name with encoder type
    config.setdefault("save_dir", "results/v0/phase3")
    config["encoder_path"] = args.encoder_path

    # Load encoder
    device = get_device()
    print(f"Loading {args.encoder_type} encoder from {args.encoder_path}")
    if args.encoder_type == "jepa":
        encoder = load_jepa_encoder(args.encoder_path, device=str(device))
    else:
        encoder = load_ae_encoder(args.encoder_path, device=str(device))

    print("=" * 60)
    print(f"Training PPO with frozen {args.encoder_type} encoder")
    print(f"  Encoder: {args.encoder_path}")
    print(f"  Seed: {config.get('seed', DEFAULT_CONFIG['seed'])}")
    print(f"  Steps: {config.get('total_timesteps', DEFAULT_CONFIG['total_timesteps']):,}")
    print("=" * 60)

    model_path = train(config=config, encoder=encoder)

    print("=" * 60)
    print(f"Training complete. Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
