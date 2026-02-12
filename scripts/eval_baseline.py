"""Evaluate a trained PPO agent and save gameplay GIFs.

Usage:
    uv run python scripts/eval_baseline.py --model results/v0/.../final_model.pt
    uv run python scripts/eval_baseline.py --model results/v0/.../final_model.pt --num-episodes 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from agents.ppo_atari import _load_checkpoint, evaluate, make_env
from env.wrappers import make_atari_env


def record_episode_gif(
    model_path: str,
    env_id: str,
    gif_path: Path,
    device: str = "cpu",
) -> float:
    """Play one episode, collect RGB frames, save as GIF, return episode reward."""
    dev = torch.device(device)
    agent, _ckpt = _load_checkpoint(model_path, dev)
    agent.eval()

    env = make_atari_env(env_id, render_mode="rgb_array", training=False)
    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    total_reward = 0.0
    done = False

    while not done:
        rgb_frame = env.render()
        frames.append(rgb_frame)

        obs_t = torch.as_tensor(obs, dtype=torch.uint8).unsqueeze(0).to(dev)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_t)

        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += float(reward)
        done = terminated or truncated

    env.close()

    pil_frames = [Image.fromarray(f) for f in frames]
    if pil_frames:
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=33,
            loop=0,
        )

    return total_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the saved model checkpoint",
    )
    parser.add_argument(
        "--env-id", type=str, default="ALE/Breakout-v5",
        help="Gymnasium environment ID (default: ALE/Breakout-v5)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10,
        help="Number of episodes for aggregate evaluation (default: 10)",
    )
    parser.add_argument(
        "--num-gifs", type=int, default=3,
        help="Number of gameplay GIFs to record (default: 3)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for inference (default: cpu)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/v0/gifs",
        help="Directory to save GIFs (default: results/v0/gifs)",
    )
    args = parser.parse_args()

    model_path = args.model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Aggregate evaluation
    # ------------------------------------------------------------------
    print(f"Evaluating {args.num_episodes} episodes on clean {args.env_id}...")

    def env_factory():
        return make_atari_env(args.env_id, training=False)

    results = evaluate(
        model_path=model_path,
        env_fn=env_factory,
        num_episodes=args.num_episodes,
        device=args.device,
    )

    print(f"Mean reward: {results['mean_reward']:.2f}")
    print(f"Std reward:  {results['std_reward']:.2f}")
    print(f"All rewards: {results['rewards']}")

    # ------------------------------------------------------------------
    # 2. Record gameplay GIFs
    # ------------------------------------------------------------------
    print(f"\nRecording {args.num_gifs} gameplay GIFs...")
    for i in range(args.num_gifs):
        gif_path = output_dir / f"episode_{i:02d}.gif"
        ep_reward = record_episode_gif(
            model_path=model_path,
            env_id=args.env_id,
            gif_path=gif_path,
            device=args.device,
        )
        print(f"  GIF {i}: reward={ep_reward:.1f} -> {gif_path}")

    print(f"\nDone. GIFs saved to {output_dir}")


if __name__ == "__main__":
    main()
