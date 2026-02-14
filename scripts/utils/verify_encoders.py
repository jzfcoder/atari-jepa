"""Verify that trained encoders produce reasonable representations.

Loads JEPA, autoencoder, and random (untrained) encoders, encodes a subset
of collected frames, and produces diagnostic plots and metrics:
  - PCA scatter plots (first 2 components) for each encoder
  - Explained variance for the first 10 PCA components
  - Nearest-neighbor grids in feature space (cosine similarity)
  - Linear probe R^2 (predicting pixel mean from 512-dim features)

Usage:
    uv run python scripts/verify_encoders.py \
        --frames results/v0/frames.npz \
        --jepa-encoder results/v0/jepa/encoder.pt \
        --ae-encoder results/v0/autoencoder/encoder.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from agents.encoder import VisionTransformer
from agents.jepa import load_jepa_encoder
from agents.autoencoder import load_ae_encoder
from agents.ppo_atari import get_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frames(frames_path: str, num_frames: int = 5000, seed: int = 42) -> np.ndarray:
    """Load a random subset of frames from an .npz file.

    Returns
    -------
    np.ndarray
        Shape (num_frames, 84, 84, 4) uint8.
    """
    data = np.load(frames_path)
    all_frames = data["frames"]
    rng = np.random.RandomState(seed)
    n = min(num_frames, len(all_frames))
    indices = rng.choice(len(all_frames), size=n, replace=False)
    indices.sort()
    return all_frames[indices]


@torch.no_grad()
def encode_frames(
    encoder: torch.nn.Module,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Encode frames into feature vectors.

    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder with forward(x) -> (batch, feature_dim). Expects input
        as (batch, channels, H, W) float32 in [0, 1].
    frames : np.ndarray
        Shape (N, 84, 84, 4) uint8.
    device : torch.device
        Device for inference.
    batch_size : int
        Batch size for encoding.

    Returns
    -------
    np.ndarray
        Shape (N, feature_dim) float32.
    """
    encoder.eval()
    features_list = []

    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        # Convert (N, 84, 84, 4) uint8 -> (N, 4, 84, 84) float32 [0, 1]
        x = torch.as_tensor(batch, dtype=torch.float32, device=device)
        x = x.permute(0, 3, 1, 2) / 255.0
        feats = encoder(x)
        features_list.append(feats.cpu().numpy())

    return np.concatenate(features_list, axis=0)


# ---------------------------------------------------------------------------
# Analysis routines
# ---------------------------------------------------------------------------

def pca_analysis(
    features: np.ndarray,
    name: str,
    output_dir: Path,
    n_components: int = 10,
) -> PCA:
    """Run PCA on features, save a 2D scatter plot, and print explained variance."""
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(features)

    # Print explained variance
    print(f"\n  {name} -- PCA explained variance (first {n_components} components):")
    for i, var in enumerate(pca.explained_variance_ratio_):
        bar = "#" * int(var * 100)
        print(f"    PC{i+1:2d}: {var:.4f}  {bar}")
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    print(f"    Cumulative (top {n_components}): {cumulative[-1]:.4f}")

    # 2D scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        components[:, 0], components[:, 1],
        c=np.arange(len(components)), cmap="viridis",
        s=3, alpha=0.5,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    ax.set_title(f"{name} -- PCA (2D)")
    plt.colorbar(scatter, ax=ax, label="frame index")
    plt.tight_layout()
    fig.savefig(str(output_dir / f"pca_{name.lower().replace(' ', '_')}.png"), dpi=150)
    plt.close(fig)

    return pca


def nearest_neighbor_grid(
    features: np.ndarray,
    frames: np.ndarray,
    name: str,
    output_dir: Path,
    n_anchors: int = 10,
    n_neighbors: int = 5,
    seed: int = 42,
) -> None:
    """For each of n_anchors random frames, find the n_neighbors nearest
    neighbors by cosine similarity and save a grid image.
    """
    rng = np.random.RandomState(seed)
    anchor_indices = rng.choice(len(features), size=n_anchors, replace=False)

    # Normalize features for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normed = features / norms

    fig, axes = plt.subplots(
        n_anchors, 1 + n_neighbors,
        figsize=(2 * (1 + n_neighbors), 2 * n_anchors),
    )
    fig.suptitle(f"{name} -- Nearest Neighbors (cosine similarity)", fontsize=12)

    for row, anchor_idx in enumerate(anchor_indices):
        # Cosine similarities to the anchor
        sims = normed @ normed[anchor_idx]
        # Exclude the anchor itself
        sims[anchor_idx] = -1.0
        neighbor_indices = np.argsort(sims)[-n_neighbors:][::-1]

        # Plot anchor
        ax = axes[row, 0]
        ax.imshow(frames[anchor_idx, :, :, 3], cmap="gray", vmin=0, vmax=255)
        ax.set_title("anchor", fontsize=7)
        ax.axis("off")

        # Plot neighbors
        for col, ni in enumerate(neighbor_indices):
            ax = axes[row, 1 + col]
            ax.imshow(frames[ni, :, :, 3], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"sim={sims[ni]:.3f}", fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(
        str(output_dir / f"neighbors_{name.lower().replace(' ', '_')}.png"),
        dpi=150,
    )
    plt.close(fig)


def linear_probe(
    features: np.ndarray,
    frames: np.ndarray,
    name: str,
) -> float:
    """Train a linear regression from 512-dim features to pixel mean.

    Uses a 80/20 train/test split. Returns the R^2 score on the test set.
    """
    # Target: mean pixel intensity of each frame (across all channels)
    targets = frames.astype(np.float32).mean(axis=(1, 2, 3))

    # Train/test split
    n = len(features)
    split = int(0.8 * n)
    X_train, X_test = features[:split], features[split:]
    y_train, y_test = targets[:split], targets[split:]

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"  {name} -- Linear probe R^2 (pixel mean): {r2:.4f}")
    return r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def verify_encoders(
    frames_path: str,
    jepa_encoder_path: str,
    ae_encoder_path: str,
    output_dir: str,
) -> None:
    """Run all verification analyses."""
    device = get_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Encoder Verification")
    print(f"  Device:  {device}")
    print(f"  Frames:  {frames_path}")
    print(f"  JEPA:    {jepa_encoder_path}")
    print(f"  AE:      {ae_encoder_path}")
    print(f"  Output:  {output_dir}")
    print("=" * 60)

    # Load frames
    print("\nLoading frames ...")
    frames = load_frames(frames_path, num_frames=5000)
    print(f"  Loaded {len(frames)} frames, shape {frames.shape}")

    # Load encoders
    print("\nLoading encoders ...")
    jepa_enc = load_jepa_encoder(jepa_encoder_path, device)
    ae_enc = load_ae_encoder(ae_encoder_path, device)

    # Random (untrained) encoder with the same architecture
    random_enc = VisionTransformer(
        in_channels=4,
        patch_size=12,
        embed_dim=192,
        num_heads=3,
        num_layers=4,
        feature_dim=512,
    ).to(device)
    random_enc.eval()

    encoders = {
        "JEPA": jepa_enc,
        "Autoencoder": ae_enc,
        "Random": random_enc,
    }

    # Encode frames with each encoder
    all_features = {}
    for name, enc in encoders.items():
        print(f"\nEncoding with {name} ...")
        feats = encode_frames(enc, frames, device)
        all_features[name] = feats
        print(f"  Feature shape: {feats.shape}, "
              f"mean={feats.mean():.4f}, std={feats.std():.4f}")

    # PCA analysis
    print("\n" + "-" * 60)
    print("PCA Analysis")
    print("-" * 60)
    for name, feats in all_features.items():
        pca_analysis(feats, name, output_path)

    # Nearest neighbor grids
    print("\n" + "-" * 60)
    print("Nearest Neighbor Analysis")
    print("-" * 60)
    for name, feats in all_features.items():
        print(f"\n  Computing nearest neighbors for {name} ...")
        nearest_neighbor_grid(feats, frames, name, output_path)
        print(f"  Saved neighbor grid for {name}.")

    # Linear probe
    print("\n" + "-" * 60)
    print("Linear Probe (pixel mean regression)")
    print("-" * 60)
    r2_scores = {}
    for name, feats in all_features.items():
        r2 = linear_probe(feats, frames, name)
        r2_scores[name] = r2

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  {'Encoder':<15} {'R^2 (pixel mean)':>20}")
    print(f"  {'-'*15} {'-'*20}")
    for name, r2 in r2_scores.items():
        print(f"  {name:<15} {r2:>20.4f}")
    print()
    print(f"All outputs saved to: {output_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify trained encoders produce reasonable representations"
    )
    parser.add_argument(
        "--frames", type=str, required=True,
        help="Path to frames .npz file (e.g. results/v0/frames.npz)",
    )
    parser.add_argument(
        "--jepa-encoder", type=str, required=True,
        help="Path to JEPA encoder checkpoint",
    )
    parser.add_argument(
        "--ae-encoder", type=str, required=True,
        help="Path to autoencoder encoder checkpoint",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/v0/encoder_analysis",
        help="Directory for output plots (default: results/v0/encoder_analysis)",
    )
    args = parser.parse_args()

    verify_encoders(
        frames_path=args.frames,
        jepa_encoder_path=args.jepa_encoder,
        ae_encoder_path=args.ae_encoder,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
