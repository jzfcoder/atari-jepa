"""Phase 3: Run all encoder-swap experiments and produce comparison tables.

Trains PPO under three encoder conditions (stock CNN, frozen JEPA, frozen AE),
each with multiple seeds, then evaluates all models under visual perturbations.

Usage:
    uv run python scripts/run_phase3.py
    uv run python scripts/run_phase3.py --config configs/phase3.yaml
    uv run python scripts/run_phase3.py --skip-training   # eval only (models must exist)
    uv run python scripts/run_phase3.py --skip-baseline    # skip stock CNN training
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from agents.ppo_atari import train, evaluate, get_device, DEFAULT_CONFIG
from agents.jepa import load_jepa_encoder
from agents.autoencoder import load_ae_encoder
from env.perturbations import make_perturbed_atari_env


# ---------------------------------------------------------------------------
# Default Phase 3 configuration
# ---------------------------------------------------------------------------

DEFAULT_PHASE3_CONFIG = {
    "env_id": "ALE/Breakout-v5",
    "total_timesteps": 10_000_000,
    "learning_rate": 2.5e-4,
    "num_envs": 8,
    "num_steps": 128,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 4,
    "update_epochs": 4,
    "clip_coef": 0.1,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "norm_adv": True,
    "capture_video": False,
    "save_interval": 500_000,
    "seeds": [1, 2, 3],
    "save_dir": "results/v0/phase3",
    "jepa_encoder_path": "results/v0/jepa/encoder_final.pt",
    "ae_encoder_path": "results/v0/autoencoder/encoder_final.pt",
    "baseline_model_path": None,
    "eval_episodes": 50,
    "eval_seeds": [42, 123, 456],
    "perturbation_levels": ["clean", "color_jitter", "noise", "mild", "hard"],
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_condition(
    condition_name: str,
    encoder,
    seed: int,
    cfg: dict,
) -> str:
    """Train PPO for one condition+seed. Returns path to saved model."""
    ppo_cfg = {
        k: v for k, v in cfg.items()
        if k in DEFAULT_CONFIG
    }
    ppo_cfg["seed"] = seed
    ppo_cfg["save_dir"] = str(Path(cfg["save_dir"]) / condition_name)
    ppo_cfg["run_name"] = f"{condition_name}_seed{seed}"

    print(f"\n{'='*60}")
    print(f"Training: {condition_name} (seed={seed})")
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
    """Format results into a comparison table.

    all_results: {condition_name: {perturbation_level: {"mean": ..., "std": ...}}}
    """
    conditions = list(all_results.keys())

    # Header
    col_w = 14
    header = f"{'Condition':<20}"
    for level in perturbation_levels:
        header += f" | {level:^{col_w}}"
    header += f" | {'Robust. Ratio':^{col_w}}"

    sep = "-" * len(header)
    lines = [sep, header, sep]

    for cond in conditions:
        row = f"{cond:<20}"
        clean_mean = all_results[cond]["clean"]["mean"]
        for level in perturbation_levels:
            m = all_results[cond][level]["mean"]
            s = all_results[cond][level]["std"]
            row += f" | {m:>6.1f} +/- {s:<4.1f}"

        # Robustness ratio: average of perturbed/clean across non-clean levels
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
    parser = argparse.ArgumentParser(
        description="Phase 3: encoder-swap experiments"
    )
    parser.add_argument(
        "--config", type=str, default="configs/phase3.yaml",
        help="Path to Phase 3 YAML config",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training, only run evaluation (models must already exist)",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip stock CNN baseline training (use existing baseline model)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for evaluation (default: auto-detect)",
    )
    args = parser.parse_args()

    # Load config
    cfg = {**DEFAULT_PHASE3_CONFIG}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg.update(yaml.safe_load(f))
    else:
        print(f"Warning: config {config_path} not found, using defaults")

    device = args.device or str(get_device())
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    seeds = cfg["seeds"]
    env_id = cfg["env_id"]

    print("=" * 60)
    print("Phase 3: Encoder-Swap Experiments")
    print(f"  Env:         {env_id}")
    print(f"  Seeds:       {seeds}")
    print(f"  Steps:       {cfg['total_timesteps']:,}")
    print(f"  Device:      {device}")
    print(f"  JEPA:        {cfg['jepa_encoder_path']}")
    print(f"  AE:          {cfg['ae_encoder_path']}")
    print(f"  Baseline:    {cfg.get('baseline_model_path', 'will train')}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    # model_paths: {condition_name: [path_seed1, path_seed2, ...]}
    model_paths: dict[str, list[str]] = {
        "Stock CNN": [],
        "JEPA": [],
        "Autoencoder": [],
    }

    if not args.skip_training:
        start_time = time.time()

        # Load encoders once
        jepa_enc = load_jepa_encoder(cfg["jepa_encoder_path"], device=device)
        ae_enc = load_ae_encoder(cfg["ae_encoder_path"], device=device)

        for seed in seeds:
            # Stock CNN baseline
            if not args.skip_baseline:
                path = train_condition("stock_cnn", None, seed, cfg)
                model_paths["Stock CNN"].append(path)

            # JEPA
            path = train_condition("jepa", jepa_enc, seed, cfg)
            model_paths["JEPA"].append(path)

            # Autoencoder
            path = train_condition("autoencoder", ae_enc, seed, cfg)
            model_paths["Autoencoder"].append(path)

        elapsed = time.time() - start_time
        print(f"\nAll training complete in {elapsed/3600:.1f} hours")

    # If skip-baseline, use the provided baseline model for all seeds
    if args.skip_baseline and cfg.get("baseline_model_path"):
        model_paths["Stock CNN"] = [cfg["baseline_model_path"]] * len(seeds)

    # If skip-training, discover existing models
    if args.skip_training:
        for cond_dir, cond_name in [("stock_cnn", "Stock CNN"), ("jepa", "JEPA"), ("autoencoder", "Autoencoder")]:
            for seed in seeds:
                run_dir = save_dir / cond_dir / f"{cond_dir}_seed{seed}"
                model_file = run_dir / "final_model.pt"
                if model_file.exists():
                    model_paths[cond_name].append(str(model_file))
        # Fall back to baseline if no stock_cnn models trained
        if not model_paths["Stock CNN"] and cfg.get("baseline_model_path"):
            model_paths["Stock CNN"] = [cfg["baseline_model_path"]] * len(seeds)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluation Phase")
    print(f"{'='*60}")

    # all_results: {condition_name: {perturbation_level: {"mean": ..., "std": ...}}}
    all_results: dict[str, dict[str, dict]] = {}

    for cond_name, paths in model_paths.items():
        if not paths:
            print(f"  Skipping {cond_name}: no models found")
            continue

        print(f"\nEvaluating {cond_name} ({len(paths)} model(s))...")
        # Aggregate across seeds
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
                # Expand: mean across eval_seeds * eval_episodes rewards
                # For aggregation, weight each model equally
                agg_by_level[lv].append(res["mean"])

        all_results[cond_name] = {}
        for lv in cfg["perturbation_levels"]:
            vals = agg_by_level[lv]
            all_results[cond_name][lv] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    table = format_comparison_table(all_results, cfg["perturbation_levels"])

    print(f"\n{'='*60}")
    print("Phase 3 Results")
    print(f"{'='*60}")
    print(table)

    # Save results
    results_path = save_dir / "phase3_results.txt"
    with open(results_path, "w") as f:
        f.write("Phase 3: Encoder-Swap Experiment Results\n")
        f.write(f"Env: {env_id}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Timesteps: {cfg['total_timesteps']:,}\n")
        f.write(f"Eval episodes: {cfg['eval_episodes']} x {len(cfg['eval_seeds'])} seeds\n\n")
        f.write(table)
        f.write("\n")

    json_path = save_dir / "phase3_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  Table: {results_path}")
    print(f"  JSON:  {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
