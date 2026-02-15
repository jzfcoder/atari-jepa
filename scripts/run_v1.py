"""V1: Unfrozen encoder experiments — training and evaluation.

Trains PPO under five encoder conditions (stock CNN, JEPA frozen, JEPA
fine-tune, AE fine-tune, ViT scratch), each with multiple seeds, then
evaluates all models under visual perturbations.

Usage:
    uv run python scripts/run_v1.py
    uv run python scripts/run_v1.py --config configs/v1.yaml
    uv run python scripts/run_v1.py --skip-training      # eval only
    uv run python scripts/run_v1.py --conditions jepa_finetune vit_scratch
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from agents.ppo_atari import train, evaluate, get_device, DEFAULT_CONFIG
from agents.encoder import VisionTransformer
from agents.jepa import load_jepa_encoder
from agents.autoencoder import load_ae_encoder
from env.perturbations import make_perturbed_atari_env


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

ALL_CONDITIONS = [
    "stock_cnn",
    "jepa_frozen",
    "jepa_finetune",
    "ae_finetune",
    "vit_scratch",
]

# Display names for results tables
DISPLAY_NAMES = {
    "stock_cnn": "Stock CNN",
    "jepa_frozen": "JEPA (frozen)",
    "jepa_finetune": "JEPA (fine-tune)",
    "ae_finetune": "AE (fine-tune)",
    "vit_scratch": "ViT (scratch)",
}


def build_encoder(condition: str, cfg: dict, device: str):
    """Return (encoder, ppo_overrides) for a given condition.

    encoder: nn.Module or None (stock CNN)
    ppo_overrides: dict of extra keys to merge into the PPO config
    """
    if condition == "stock_cnn":
        return None, {}

    elif condition == "jepa_frozen":
        enc = load_jepa_encoder(cfg["jepa_encoder_path"], device=device)
        return enc, {"freeze_encoder": True}

    elif condition == "jepa_finetune":
        enc = load_jepa_encoder(cfg["jepa_encoder_path"], device=device)
        return enc, {
            "freeze_encoder": False,
            "encoder_lr_scale": cfg.get("encoder_lr_scale", 0.1),
        }

    elif condition == "ae_finetune":
        enc = load_ae_encoder(cfg["ae_encoder_path"], device=device)
        return enc, {
            "freeze_encoder": False,
            "encoder_lr_scale": cfg.get("encoder_lr_scale", 0.1),
        }

    elif condition == "vit_scratch":
        # Random-init ViT with same architecture as the pretrained encoders.
        # Read arch params from the JEPA config if available.
        enc = VisionTransformer(
            in_channels=4,
            patch_size=12,
            embed_dim=192,
            num_heads=3,
            num_layers=4,
            feature_dim=512,
        )
        return enc, {
            "freeze_encoder": False,
            "encoder_lr_scale": 1.0,  # full LR — no pretrained weights to protect
        }

    else:
        raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_condition(
    condition: str,
    encoder,
    seed: int,
    cfg: dict,
    ppo_overrides: dict,
) -> str:
    """Train PPO for one condition+seed. Returns path to saved model."""
    ppo_cfg = {k: v for k, v in cfg.items() if k in DEFAULT_CONFIG}
    ppo_cfg.update(ppo_overrides)
    ppo_cfg["seed"] = seed
    ppo_cfg["save_dir"] = str(Path(cfg["save_dir"]) / condition)
    ppo_cfg["run_name"] = f"{condition}_seed{seed}"

    print(f"\n{'='*60}")
    print(f"Training: {DISPLAY_NAMES.get(condition, condition)} (seed={seed})")
    print(f"{'='*60}")

    return train(config=ppo_cfg, encoder=encoder)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    env_id: str,
    perturbation_levels: list[str],
    num_episodes: int,
    eval_seeds: list[int],
    device: str,
) -> dict[str, dict]:
    """Evaluate one model under all perturbation levels."""
    results = {}
    for level in perturbation_levels:
        all_rewards = []
        for seed in eval_seeds:
            def env_fn(_lvl=level, _seed=seed):
                return make_perturbed_atari_env(
                    env_id, perturbation_level=_lvl, seed=_seed, training=False,
                )
            res = evaluate(
                model_path=model_path,
                env_fn=env_fn,
                num_episodes=num_episodes,
                device=device,
            )
            all_rewards.extend(res["rewards"])

        results[level] = {
            "mean": float(np.mean(all_rewards)),
            "std": float(np.std(all_rewards)),
        }
    return results


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def format_comparison_table(
    all_results: dict[str, dict[str, dict]],
    perturbation_levels: list[str],
) -> str:
    """Format results into a comparison table."""
    col_w = 14
    header = f"{'Condition':<22}"
    for level in perturbation_levels:
        header += f" | {level:^{col_w}}"
    header += f" | {'Robust. Ratio':^{col_w}}"

    sep = "-" * len(header)
    lines = [sep, header, sep]

    for cond, display in DISPLAY_NAMES.items():
        if cond not in all_results:
            continue
        row = f"{display:<22}"
        clean_mean = all_results[cond]["clean"]["mean"]
        for level in perturbation_levels:
            m = all_results[cond][level]["mean"]
            s = all_results[cond][level]["std"]
            row += f" | {m:>6.1f} +/- {s:<4.1f}"

        if clean_mean > 0:
            ratios = [
                all_results[cond][lv]["mean"] / clean_mean
                for lv in perturbation_levels if lv != "clean"
            ]
            avg_ratio = float(np.mean(ratios))
        else:
            avg_ratio = 0.0
        row += f" | {avg_ratio:^{col_w}.3f}"
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="V1: unfrozen encoder experiments")
    parser.add_argument(
        "--config", type=str, default="configs/v1.yaml",
        help="Path to V1 YAML config",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training, only run evaluation (models must already exist)",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        choices=ALL_CONDITIONS,
        help="Run only these conditions (default: all from config)",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load config
    cfg = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config not found: {config_path}")

    device = args.device or str(get_device())
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    seeds = cfg["seeds"]
    env_id = cfg["env_id"]
    conditions = args.conditions or cfg.get("conditions", ALL_CONDITIONS)

    print("=" * 60)
    print("V1: Unfrozen Encoder Experiments")
    print(f"  Env:          {env_id}")
    print(f"  Seeds:        {seeds}")
    print(f"  Steps:        {cfg['total_timesteps']:,}")
    print(f"  Device:       {device}")
    print(f"  Conditions:   {conditions}")
    print(f"  LR scale:     {cfg.get('encoder_lr_scale', 0.1)}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    # model_paths: {condition: [path_seed1, path_seed2, ...]}
    model_paths: dict[str, list[str]] = {c: [] for c in conditions}

    if not args.skip_training:
        start_time = time.time()

        for condition in conditions:
            for seed in seeds:
                # Build a fresh encoder for each seed (important for vit_scratch
                # so each seed gets independent random init)
                encoder, ppo_overrides = build_encoder(condition, cfg, device)
                path = train_condition(condition, encoder, seed, cfg, ppo_overrides)
                model_paths[condition].append(path)

        elapsed = time.time() - start_time
        print(f"\nAll training complete in {elapsed/3600:.1f} hours")
    else:
        # Discover existing models
        for condition in conditions:
            for seed in seeds:
                model_file = save_dir / condition / f"{condition}_seed{seed}" / "final_model.pt"
                if model_file.exists():
                    model_paths[condition].append(str(model_file))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluation Phase")
    print(f"{'='*60}")

    all_results: dict[str, dict[str, dict]] = {}

    for condition in conditions:
        paths = model_paths[condition]
        if not paths:
            print(f"  Skipping {condition}: no models found")
            continue

        display = DISPLAY_NAMES.get(condition, condition)
        print(f"\nEvaluating {display} ({len(paths)} model(s))...")

        agg_by_level: dict[str, list[float]] = {
            lv: [] for lv in cfg["perturbation_levels"]
        }

        for model_path in paths:
            print(f"  Model: {model_path}")
            results = evaluate_model(
                model_path=model_path,
                env_id=env_id,
                perturbation_levels=cfg["perturbation_levels"],
                num_episodes=cfg["eval_episodes"],
                eval_seeds=cfg["eval_seeds"],
                device=device,
            )
            for lv, res in results.items():
                agg_by_level[lv].append(res["mean"])

        all_results[condition] = {}
        for lv in cfg["perturbation_levels"]:
            vals = agg_by_level[lv]
            all_results[condition][lv] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    table = format_comparison_table(all_results, cfg["perturbation_levels"])

    print(f"\n{'='*60}")
    print("V1 Results")
    print(f"{'='*60}")
    print(table)

    results_path = save_dir / "v1_results.txt"
    with open(results_path, "w") as f:
        f.write("V1: Unfrozen Encoder Experiment Results\n")
        f.write(f"Env: {env_id}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Timesteps: {cfg['total_timesteps']:,}\n")
        f.write(f"Encoder LR scale: {cfg.get('encoder_lr_scale', 0.1)}\n")
        f.write(f"Eval episodes: {cfg['eval_episodes']} x {len(cfg['eval_seeds'])} seeds\n\n")
        f.write(table)
        f.write("\n")

    json_path = save_dir / "v1_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  Table: {results_path}")
    print(f"  JSON:  {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
