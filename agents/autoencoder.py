"""Autoencoder baseline for the JEPA encoder-swap experiment.

This module trains a VisionTransformer encoder with a pixel-reconstruction
objective (MSE loss).  It serves as a control condition: if the JEPA encoder
improves visual robustness but the autoencoder does not, we can attribute the
benefit specifically to JEPA's representation-learning objective rather than
to the ViT architecture or the extra pre-training data.

The trained encoder checkpoint is saved in the same format as the JEPA encoder
so it can be loaded and plugged directly into the PPO Agent.

Usage:
    uv run python -m agents.autoencoder                    # train with defaults
    uv run python -m agents.autoencoder --frames_path ...  # override a config key
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from agents.encoder import TransformerBlock, VisionTransformer
from agents.ppo_atari import get_device


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_AE_CONFIG = {
    "frames_path": "results/v0/frames.npz",
    "in_channels": 4,
    "patch_size": 12,
    "embed_dim": 192,
    "num_heads": 3,
    "num_layers": 4,
    "feature_dim": 512,
    "decoder_embed_dim": 96,
    "decoder_num_heads": 2,
    "decoder_num_layers": 2,
    "learning_rate": 1.5e-4,
    "weight_decay": 0.05,
    "epochs": 100,
    "batch_size": 256,
    "seed": 42,
    "save_dir": "results/v0/autoencoder",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FrameDataset(Dataset):
    """Load pre-collected Atari frames for self-supervised training.

    Expected .npz layout:
        key "frames", shape (N, 84, 84, 4), dtype uint8

    Each sample is returned as a float32 tensor of shape (4, 84, 84) in [0, 1].
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        # (N, 84, 84, 4) uint8
        self.frames = data["frames"]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.frames[idx]                       # (84, 84, 4) uint8
        tensor = torch.from_numpy(frame).float() / 255.0  # [0, 1]
        tensor = tensor.permute(2, 0, 1)               # (4, 84, 84)
        return tensor


# ---------------------------------------------------------------------------
# Patch Decoder
# ---------------------------------------------------------------------------

class PatchDecoder(nn.Module):
    """Small transformer decoder that reconstructs the original image from
    per-patch embeddings produced by the VisionTransformer encoder.

    Pipeline:
        (batch, num_patches, encoder_embed_dim)
        -> linear projection to decoder_embed_dim
        -> add learnable position embeddings
        -> N transformer blocks
        -> linear projection to (patch_size * patch_size * in_channels)
        -> reshape / unpatchify to (batch, in_channels, H, W)
    """

    def __init__(
        self,
        num_patches: int = 49,
        patch_size: int = 12,
        in_channels: int = 4,
        encoder_embed_dim: int = 192,
        decoder_embed_dim: int = 96,
        num_heads: int = 2,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Grid dimensions (84 / 12 = 7)
        self.grid_h = int(math.isqrt(num_patches))
        self.grid_w = self.grid_h
        assert self.grid_h * self.grid_w == num_patches, (
            f"num_patches={num_patches} is not a perfect square"
        )

        # Input projection: encoder dim -> decoder dim
        self.input_proj = nn.Linear(encoder_embed_dim, decoder_embed_dim)

        # Learnable position embeddings for the decoder
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks (reuse the same TransformerBlock from encoder.py)
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(decoder_embed_dim)

        # Output projection: decoder dim -> pixels per patch
        self.output_proj = nn.Linear(
            decoder_embed_dim, patch_size * patch_size * in_channels
        )

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Reconstruct an image from per-patch embeddings.

        Args:
            patch_embeddings: (batch, num_patches, encoder_embed_dim)

        Returns:
            Reconstructed image: (batch, in_channels, 84, 84)
        """
        B = patch_embeddings.shape[0]

        # Project to decoder dimension and add positional information
        x = self.input_proj(patch_embeddings)       # (B, N, decoder_dim)
        x = x + self.pos_embed                      # (B, N, decoder_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Project each patch token to pixel space
        x = self.output_proj(x)  # (B, N, patch_size*patch_size*in_channels)

        # Unpatchify: rearrange back to spatial image
        # (B, N, P*P*C) -> (B, grid_h, grid_w, P, P, C)
        x = x.reshape(
            B,
            self.grid_h, self.grid_w,
            self.patch_size, self.patch_size,
            self.in_channels,
        )
        # (B, grid_h, grid_w, P, P, C) -> (B, C, grid_h*P, grid_w*P)
        x = x.permute(0, 5, 1, 3, 2, 4)            # (B, C, gh, P, gw, P)
        x = x.reshape(
            B,
            self.in_channels,
            self.grid_h * self.patch_size,
            self.grid_w * self.patch_size,
        )
        return x  # (B, 4, 84, 84)


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """VisionTransformer encoder + PatchDecoder with MSE reconstruction loss."""

    def __init__(self, encoder: VisionTransformer, decoder: PatchDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with reconstruction loss.

        Args:
            x: (batch, 4, 84, 84) float32 in [0, 1]

        Returns:
            loss:          scalar MSE reconstruction loss
            reconstructed: (batch, 4, 84, 84) reconstructed image
        """
        patch_emb = self.encoder.forward_features(x)   # (B, 49, 192)
        reconstructed = self.decoder(patch_emb)         # (B, 4, 84, 84)
        loss = F.mse_loss(reconstructed, x)
        return loss, reconstructed


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_autoencoder(config: dict | None = None) -> str:
    """Train the autoencoder and save the encoder checkpoint.

    Args:
        config: Training configuration dict.  Missing keys are filled from
                DEFAULT_AE_CONFIG.

    Returns:
        Path to the saved encoder checkpoint file.
    """
    cfg = {**DEFAULT_AE_CONFIG, **(config or {})}

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = get_device()
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Save directory: {save_dir}")

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------
    writer = SummaryWriter(str(save_dir / "tb"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(
            f"|{k}|{v}|" for k, v in cfg.items()
        ),
    )

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    dataset = FrameDataset(cfg["frames_path"])
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Loaded {len(dataset)} frames from {cfg['frames_path']}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    encoder = VisionTransformer(
        in_channels=cfg["in_channels"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        feature_dim=cfg["feature_dim"],
    )

    # Derive num_patches from the encoder's grid (84 // 12 = 7, 7*7 = 49)
    grid_size = 84 // cfg["patch_size"]
    num_patches = grid_size * grid_size

    decoder = PatchDecoder(
        num_patches=num_patches,
        patch_size=cfg["patch_size"],
        in_channels=cfg["in_channels"],
        encoder_embed_dim=cfg["embed_dim"],
        decoder_embed_dim=cfg["decoder_embed_dim"],
        num_heads=cfg["decoder_num_heads"],
        num_layers=cfg["decoder_num_layers"],
    )

    model = Autoencoder(encoder, decoder).to(device)

    num_params_enc = sum(p.numel() for p in encoder.parameters())
    num_params_dec = sum(p.numel() for p in decoder.parameters())
    print(
        f"Encoder params: {num_params_enc:,} | "
        f"Decoder params: {num_params_dec:,} | "
        f"Total: {num_params_enc + num_params_dec:,}"
    )

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    total_steps = cfg["epochs"] * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    start_time = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(device)  # (B, 4, 84, 84)

            loss, _ = model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            num_batches += 1

            # Per-step logging
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar(
                "train/lr", optimizer.param_groups[0]["lr"], global_step
            )

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        print(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"loss={avg_loss:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"elapsed={elapsed:.0f}s"
        )

        # Periodic checkpoint (every 10 epochs)
        if epoch % 10 == 0:
            ckpt_path = save_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": avg_loss,
                },
                ckpt_path,
            )
            print(f"  [checkpoint] saved to {ckpt_path}")

    # ------------------------------------------------------------------
    # Final save (encoder-only format, compatible with JEPA loader)
    # ------------------------------------------------------------------
    final_path = save_dir / "encoder_final.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "config": cfg,
            "epoch": cfg["epochs"],
        },
        final_path,
    )
    print(f"Training complete. Encoder saved to {final_path}")

    writer.close()
    return str(final_path)


# ---------------------------------------------------------------------------
# Loading a trained encoder
# ---------------------------------------------------------------------------

def load_ae_encoder(
    checkpoint_path: str, device: str = "cpu"
) -> VisionTransformer:
    """Load a trained autoencoder encoder from a checkpoint.

    The returned VisionTransformer is ready to be plugged into the PPO Agent::

        encoder = load_ae_encoder("results/v0/autoencoder/encoder_final.pt")
        agent = Agent(num_actions, encoder=encoder)

    Args:
        checkpoint_path: Path to a .pt checkpoint saved by train_autoencoder.
        device: Target device string ("cpu", "cuda", "mps").

    Returns:
        VisionTransformer with loaded weights, in eval mode.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    encoder = VisionTransformer(
        in_channels=cfg["in_channels"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        feature_dim=cfg["feature_dim"],
    )
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> dict:
    """Parse command-line arguments and return a config dict override."""
    parser = argparse.ArgumentParser(
        description="Train autoencoder baseline for JEPA comparison"
    )
    for key, default in DEFAULT_AE_CONFIG.items():
        arg_type = type(default)
        parser.add_argument(f"--{key}", type=arg_type, default=None)
    args = parser.parse_args()

    overrides = {}
    for key in DEFAULT_AE_CONFIG:
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val
    return overrides


if __name__ == "__main__":
    overrides = _parse_args()
    train_autoencoder(overrides)
