"""Collect gameplay frames for self-supervised encoder training.

Collects ~100K observations from an Atari environment using either a random
policy or a mix of random + trained PPO policy. Frames are saved as a .npz
file with key "frames", shape (N, 84, 84, 4) uint8.

Usage:
    uv run python scripts/collect_frames.py
    uv run python scripts/collect_frames.py --num-frames 50000 --model results/v0/.../final_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from env.wrappers import make_atari_env
from agents.ppo_atari import _load_checkpoint, get_device


def collect_frames(
    env_id: str,
    num_frames: int,
    output: str,
    model_path: str | None = None,
) -> None:
    """Collect gameplay frames and save to .npz.

    Parameters
    ----------
    env_id : str
        Gymnasium environment identifier (e.g. "ALE/Breakout-v5").
    num_frames : int
        Target number of frames to collect.
    output : str
        Path to save the output .npz file.
    model_path : str | None
        Optional path to a PPO checkpoint. If provided, 50% of actions
        come from the trained policy and 50% from random sampling.
    """
    device = get_device()

    # Create env with training=False so we get full episodes (no EpisodicLife
    # or ClipReward wrappers).
    env = make_atari_env(env_id, seed=42, training=False)
    num_actions = env.action_space.n

    # Optionally load a trained PPO agent for mixed policy collection.
    agent = None
    if model_path is not None:
        print(f"Loading PPO checkpoint from {model_path}")
        agent, _ = _load_checkpoint(model_path, device)
        agent.eval()

    # We save every 4th frame to reduce temporal correlation.
    save_interval = 4

    frames: list[np.ndarray] = []
    step_count = 0
    episodes_completed = 0

    obs, _ = env.reset()

    print(f"Collecting {num_frames} frames from {env_id} ...")
    policy_desc = "50% random + 50% trained" if agent is not None else "100% random"
    print(f"Policy: {policy_desc}")

    while len(frames) < num_frames:
        # Choose action: mixed policy or pure random.
        if agent is not None and step_count % 2 == 0:
            # Trained policy action
            obs_t = torch.as_tensor(obs, dtype=torch.uint8).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
            action = action.item()
        else:
            # Random action
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        # Save every 4th frame.
        if step_count % save_interval == 0:
            frames.append(obs.copy())

        if terminated or truncated:
            obs, _ = env.reset()
            episodes_completed += 1

        # Progress report every 10000 collected frames.
        if len(frames) % 10000 == 0 and len(frames) > 0:
            print(f"  collected {len(frames)}/{num_frames} frames "
                  f"({episodes_completed} episodes)")

    env.close()

    # Stack into a single array: (N, 84, 84, 4) uint8
    frames_arr = np.array(frames[:num_frames], dtype=np.uint8)

    # Save .npz
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), frames=frames_arr)

    # Print stats
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print("=" * 60)
    print(f"Collection complete.")
    print(f"  Frames collected: {frames_arr.shape[0]}")
    print(f"  Frame shape:      {frames_arr.shape[1:]}")
    print(f"  Total steps:      {step_count}")
    print(f"  Episodes:         {episodes_completed}")
    print(f"  File size:        {file_size_mb:.1f} MB")
    print(f"  Saved to:         {output_path}")
    print("=" * 60)

    # Save 16 sample frames as a grid PNG for visual inspection.
    _save_sample_grid(frames_arr, output_path)


def _save_sample_grid(frames: np.ndarray, output_path: Path) -> None:
    """Save a 4x4 grid of sample frames as a PNG image.

    Each frame has shape (84, 84, 4) with 4 stacked grayscale channels.
    We display only the most recent channel (index 3) for each sample.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_samples = 16
    # Pick evenly spaced indices across the dataset.
    indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Sample Collected Frames (most recent channel)", fontsize=12)

    for idx, ax in zip(indices, axes.flat):
        # Show the most recent frame in the 4-frame stack (channel 3).
        ax.imshow(frames[idx, :, :, 3], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"#{idx}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    grid_path = output_path.parent / "sample_frames.png"
    fig.savefig(str(grid_path), dpi=150)
    plt.close(fig)
    print(f"  Sample grid saved to: {grid_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Atari gameplay frames for self-supervised training"
    )
    parser.add_argument(
        "--env-id", type=str, default="ALE/Breakout-v5",
        help="Gymnasium environment ID (default: ALE/Breakout-v5)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=100000,
        help="Number of frames to collect (default: 100000)",
    )
    parser.add_argument(
        "--output", type=str, default="results/v0/frames.npz",
        help="Output path for the .npz file (default: results/v0/frames.npz)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Optional path to a partially-trained PPO checkpoint. "
             "If provided, uses 50%% random + 50%% trained policy.",
    )
    args = parser.parse_args()

    collect_frames(
        env_id=args.env_id,
        num_frames=args.num_frames,
        output=args.output,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
