"""Visualize input gradient saliency maps for each encoder type.

Loads stock CNN, JEPA, and AE models, computes gradient-based saliency
(backprop value output w.r.t. input pixels), and produces side-by-side
comparison images.

Usage:
    uv run python scripts/visualize_saliency.py
    uv run python scripts/visualize_saliency.py \
        --stock-cnn results/v0/ppo_Breakout-v5_42_1770766846/final_model.pt \
        --jepa results/v0/phase3/jepa/jepa_seed1/final_model.pt \
        --ae results/v0/phase3/autoencoder/autoencoder_seed1/final_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.ppo_atari import _load_checkpoint, get_device


def compute_saliency(agent, obs_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Compute input gradient saliency for the value function.

    Args:
        agent: PPO Agent with encode() and critic.
        obs_tensor: (1, 84, 84, 4) uint8 tensor.
        device: Torch device.

    Returns:
        Saliency map as (84, 84) float32 numpy array, normalized to [0, 1].
    """
    obs = obs_tensor.clone().to(device).float()
    # Preprocess: (1, 84, 84, 4) -> (1, 4, 84, 84) float [0, 1]
    obs = obs.permute(0, 3, 1, 2) / 255.0
    obs.requires_grad_(True)

    # Forward through encoder + critic
    features = agent.encoder(obs)
    value = agent.critic(features)
    value.backward()

    # Gradient w.r.t. input: take abs and max across channels
    grad = obs.grad.detach().cpu().numpy()[0]  # (4, 84, 84)
    saliency = np.abs(grad).max(axis=0)  # (84, 84)

    # Normalize to [0, 1]
    smax = saliency.max()
    if smax > 0:
        saliency = saliency / smax

    return saliency


def load_sample_frames(frames_path: str, num_frames: int = 4, seed: int = 42) -> np.ndarray:
    """Load a few representative frames from the collected frames."""
    data = np.load(frames_path)
    frames = data["frames"]  # (N, 84, 84, 4) uint8

    rng = np.random.RandomState(seed)
    # Choose frames spread across the dataset
    n = len(frames)
    indices = np.linspace(n * 0.1, n * 0.9, num_frames, dtype=int)
    # Add some randomness
    indices = np.clip(indices + rng.randint(-100, 100, size=num_frames), 0, n - 1)
    return frames[indices]


def plot_saliency_grid(
    frames: np.ndarray,
    agents: dict[str, object],
    device: torch.device,
    output_dir: Path,
) -> None:
    """Create a grid: rows = frames, cols = original + saliency per encoder."""
    n_frames = len(frames)
    encoder_names = list(agents.keys())
    n_cols = 1 + len(encoder_names)  # original + one per encoder

    fig, axes = plt.subplots(
        n_frames, n_cols,
        figsize=(3 * n_cols, 3 * n_frames),
    )
    if n_frames == 1:
        axes = axes[np.newaxis, :]

    # Column headers
    col_titles = ["Original Frame"] + [f"{name} Saliency" for name in encoder_names]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight="bold")

    for i in range(n_frames):
        frame = frames[i]  # (84, 84, 4) uint8
        obs_tensor = torch.as_tensor(frame, dtype=torch.uint8).unsqueeze(0)

        # Original frame (last channel = most recent)
        axes[i, 0].imshow(frame[:, :, 3], cmap="gray", vmin=0, vmax=255)
        axes[i, 0].axis("off")

        # Saliency for each encoder
        for j, (name, agent) in enumerate(agents.items()):
            saliency = compute_saliency(agent, obs_tensor, device)
            axes[i, j + 1].imshow(saliency, cmap="hot", vmin=0, vmax=1)
            axes[i, j + 1].axis("off")

    plt.suptitle("Input Gradient Saliency (Value Function)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = output_dir / "saliency_comparison.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize encoder saliency maps")
    parser.add_argument(
        "--stock-cnn", type=str,
        default="results/v0/ppo_Breakout-v5_42_1770766846/final_model.pt",
        help="Path to stock CNN model checkpoint",
    )
    parser.add_argument(
        "--jepa", type=str,
        default="results/v0/phase3/jepa/jepa_seed1/final_model.pt",
        help="Path to JEPA PPO model checkpoint",
    )
    parser.add_argument(
        "--ae", type=str,
        default="results/v0/phase3/autoencoder/autoencoder_seed1/final_model.pt",
        help="Path to Autoencoder PPO model checkpoint",
    )
    parser.add_argument(
        "--frames", type=str, default="results/v0/frames.npz",
        help="Path to collected frames .npz file",
    )
    parser.add_argument(
        "--num-frames", type=int, default=4,
        help="Number of frames to visualize",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/v0/phase4",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    # Load models
    print("Loading models...")
    agents = {}
    for name, path in [("Stock CNN", args.stock_cnn), ("JEPA", args.jepa), ("Autoencoder", args.ae)]:
        if not Path(path).exists():
            print(f"  Warning: {path} not found, skipping {name}")
            continue
        agent, _ = _load_checkpoint(path, device)
        agent.eval()
        agents[name] = agent
        print(f"  Loaded {name} from {path}")

    if not agents:
        print("No models found. Exiting.")
        return

    # Load frames
    print(f"Loading frames from {args.frames}...")
    frames = load_sample_frames(args.frames, num_frames=args.num_frames)
    print(f"  Using {len(frames)} frames")

    # Plot
    print("Computing saliency maps...")
    plot_saliency_grid(frames, agents, device, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
