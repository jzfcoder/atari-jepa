"""Verify that the Atari environment works and save sample frames.

Usage:
    uv run python scripts/test_env.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium
import numpy as np
from PIL import Image

from env.wrappers import (
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    make_atari_env,
)
from env.perturbations import ColorJitterWrapper, GaussianNoiseWrapper


def save_frame(obs: np.ndarray, path: Path) -> None:
    """Save a single grayscale frame (the latest from the frame stack) as PNG.

    The observation has shape (84, 84, 4) where the last axis is the frame
    stack.  We take the last channel (most recent frame).
    """
    frame = obs[:, :, -1]  # (84, 84), uint8
    img = Image.fromarray(frame, mode="L")
    img.save(path)


def make_rgb_env(env_id: str, seed: int = 42) -> gymnasium.Env:
    """Create a minimal Atari env that stays in RGB (no grayscale/framestack).

    Used for capturing visual comparison frames.
    """
    env = gymnasium.make(env_id, render_mode="rgb_array")
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env.reset(seed=seed)
    return env


def capture_obs_after_steps(env: gymnasium.Env, steps: int = 50) -> np.ndarray:
    """Take random actions and return the last observation (RGB)."""
    obs = None
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
    return obs


def save_comparison(clean_frame: np.ndarray, mild_frame: np.ndarray,
                    hard_frame: np.ndarray, path: Path) -> None:
    """Save clean / mild / hard RGB frames side by side as a single PNG."""
    clean_img = Image.fromarray(clean_frame)
    mild_img = Image.fromarray(mild_frame)
    hard_img = Image.fromarray(hard_frame)

    h = max(clean_img.height, mild_img.height, hard_img.height)
    w_total = clean_img.width + mild_img.width + hard_img.width
    canvas = Image.new("RGB", (w_total, h))
    canvas.paste(clean_img, (0, 0))
    canvas.paste(mild_img, (clean_img.width, 0))
    canvas.paste(hard_img, (clean_img.width + mild_img.width, 0))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Atari environment setup")
    parser.add_argument("--env-id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="results/v0/sample_frames")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Preprocessed environment -- random actions, save sample frames
    # ------------------------------------------------------------------
    print(f"Creating preprocessed environment: {args.env_id}")
    env = make_atari_env(args.env_id, seed=42)

    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    observations: list[np.ndarray] = [obs]
    rewards: list[float] = []
    for _ in range(args.num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        rewards.append(float(reward))
        if terminated or truncated:
            obs, _ = env.reset()
            observations.append(obs)

    total_reward = sum(rewards)
    print(f"Total reward over {args.num_steps} random steps: {total_reward}")
    print(f"Non-zero rewards: {sum(1 for r in rewards if r != 0)}")

    # Save 5 evenly-spaced grayscale observation frames.
    indices = np.linspace(0, len(observations) - 1, 5, dtype=int)
    for i, idx in enumerate(indices):
        frame_path = output_dir / f"frame_{i:02d}.png"
        save_frame(observations[idx], frame_path)
        print(f"Saved {frame_path}")

    env.close()

    # ------------------------------------------------------------------
    # 2. RGB comparison: clean vs mild vs hard perturbations
    # ------------------------------------------------------------------
    print("\nCapturing RGB comparison frames...")

    # Clean
    clean_env = make_rgb_env(args.env_id, seed=42)
    clean_obs = capture_obs_after_steps(clean_env, steps=50)
    clean_env.close()

    # Mild perturbation
    mild_env = make_rgb_env(args.env_id, seed=42)
    mild_env = ColorJitterWrapper(mild_env, hue_range=0.2, sat_range=0.3, bright_range=0.2)
    mild_env = GaussianNoiseWrapper(mild_env, std=10.0)
    mild_env.reset(seed=42)
    mild_obs = capture_obs_after_steps(mild_env, steps=50)
    mild_env.close()

    # Hard perturbation
    hard_env = make_rgb_env(args.env_id, seed=42)
    hard_env = ColorJitterWrapper(hard_env, hue_range=0.4, sat_range=0.5, bright_range=0.4)
    hard_env = GaussianNoiseWrapper(hard_env, std=25.0)
    hard_env.reset(seed=42)
    hard_obs = capture_obs_after_steps(hard_env, steps=50)
    hard_env.close()

    comparison_path = output_dir / "comparison_clean_mild_hard.png"
    save_comparison(clean_obs, mild_obs, hard_obs, comparison_path)
    print(f"Saved comparison image: {comparison_path}")

    print(f"\nDone. All frames saved to {output_dir}")


if __name__ == "__main__":
    main()
