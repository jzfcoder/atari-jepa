"""JEPA (Joint Embedding Predictive Architecture) for Atari frames.

Trains a ViT encoder self-supervised via masked patch prediction in
representation space. The encoder can then be frozen and plugged into
the PPO Agent for visually robust RL.
"""

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from agents.encoder import VisionTransformer, TransformerBlock, trunc_normal_
from agents.ppo_atari import get_device


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_JEPA_CONFIG = {
    "frames_path": "results/v0/frames.npz",
    "in_channels": 4,
    "patch_size": 12,
    "embed_dim": 192,
    "num_heads": 3,
    "num_layers": 4,
    "feature_dim": 512,
    "predictor_embed_dim": 96,
    "predictor_num_heads": 2,
    "predictor_num_layers": 2,
    "mask_ratio": 0.5,
    "ema_decay": 0.996,
    "learning_rate": 1.5e-4,
    "weight_decay": 0.05,
    "epochs": 100,
    "batch_size": 256,
    "seed": 42,
    "save_dir": "results/v0/jepa",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FrameDataset(Dataset):
    """Load pre-collected Atari frames for self-supervised training.

    Expects a .npz file with key "frames", shape (N, 84, 84, 4) uint8.
    Returns float32 tensors (4, 84, 84) in [0, 1].
    """

    def __init__(self, path):
        data = np.load(path)
        self.frames = data["frames"]  # (N, 84, 84, 4)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]  # (84, 84, 4) uint8
        x = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        return x


# ---------------------------------------------------------------------------
# Block masking
# ---------------------------------------------------------------------------

def generate_block_mask(batch_size, grid_size=7, mask_ratio=0.5, device="cpu"):
    """Generate block masks for a batch.

    For each sample, pick a random rectangular block in the 7x7 patch grid,
    then pad or trim to ensure exactly `target_masked` patches are masked
    in every sample (required for batched gather operations).

    Returns bool tensor (B, num_patches), True = MASKED.
    """
    num_patches = grid_size * grid_size
    target_masked = max(1, int(num_patches * mask_ratio))
    masks = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

    for i in range(batch_size):
        # Sample block dimensions to approximately hit target_masked
        bh = random.randint(2, min(5, grid_size))
        bw = max(1, min(target_masked // bh, grid_size))
        bh = min(bh, grid_size)
        bw = min(bw, grid_size)

        # Random top-left corner
        r = random.randint(0, grid_size - bh)
        c = random.randint(0, grid_size - bw)

        for dr in range(bh):
            for dc in range(bw):
                masks[i, (r + dr) * grid_size + (c + dc)] = True

        # Enforce exactly target_masked patches per sample
        current = int(masks[i].sum().item())
        if current < target_masked:
            # Randomly mask additional unmasked patches
            unmasked = (~masks[i]).nonzero(as_tuple=False).squeeze(-1)
            extra = unmasked[torch.randperm(len(unmasked), device=device)[:target_masked - current]]
            masks[i, extra] = True
        elif current > target_masked:
            # Randomly unmask some patches
            masked = masks[i].nonzero(as_tuple=False).squeeze(-1)
            drop = masked[torch.randperm(len(masked), device=device)[:current - target_masked]]
            masks[i, drop] = False

    return masks


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class JEPAPredictor(nn.Module):
    """Small transformer that predicts target patch representations."""

    def __init__(self, num_patches=49, encoder_embed_dim=192,
                 predictor_embed_dim=96, num_heads=2, num_layers=2):
        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.predictor_embed_dim = predictor_embed_dim

        self.input_proj = nn.Linear(encoder_embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.output_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.mask_token)
        trunc_normal_(self.pos_embed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, context_emb, context_idx, target_idx):
        """Predict representations for target patches.

        Args:
            context_emb: (B, num_visible, encoder_embed_dim)
            context_idx: (B, num_visible) int — patch indices of visible patches
            target_idx:  (B, num_target) int — patch indices of masked patches

        Returns:
            (B, num_target, encoder_embed_dim) predicted target representations
        """
        B = context_emb.shape[0]
        num_visible = context_emb.shape[1]
        num_target = target_idx.shape[1]

        # Project context to predictor dim
        ctx = self.input_proj(context_emb)  # (B, V, pred_D)

        # Add position embeddings for context patches
        ctx_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1), 1,
            context_idx.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim)
        )
        ctx = ctx + ctx_pos

        # Create mask tokens for targets with position embeddings
        mask_tokens = self.mask_token.expand(B, num_target, -1)
        tgt_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1), 1,
            target_idx.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim)
        )
        mask_tokens = mask_tokens + tgt_pos

        # Concatenate context + mask tokens and run through transformer
        tokens = torch.cat([ctx, mask_tokens], dim=1)  # (B, V+T, pred_D)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # Extract only target predictions (last num_target tokens)
        target_preds = tokens[:, num_visible:]  # (B, T, pred_D)
        return self.output_proj(target_preds)   # (B, T, enc_D)


# ---------------------------------------------------------------------------
# JEPA
# ---------------------------------------------------------------------------

class JEPA(nn.Module):
    """Full JEPA system: context encoder + EMA target encoder + predictor."""

    def __init__(self, context_encoder, predictor, ema_decay=0.996):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = copy.deepcopy(context_encoder)
        self.predictor = predictor
        self.ema_decay = ema_decay

        # Target encoder never gets gradients
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update: target = decay * target + (1-decay) * context."""
        for tp, cp in zip(self.target_encoder.parameters(),
                          self.context_encoder.parameters()):
            tp.data.mul_(self.ema_decay).add_(cp.data, alpha=1.0 - self.ema_decay)

    def forward(self, x, mask):
        """Compute JEPA loss.

        Args:
            x:    (B, 4, 84, 84) float32.
            mask: (B, num_patches) bool, True = MASKED.

        Returns:
            loss: scalar, L2 loss between predicted and target representations.
        """
        B = x.shape[0]
        num_patches = mask.shape[1]
        visible = ~mask

        num_visible = int(visible[0].sum().item())
        num_target = int(mask[0].sum().item())

        # Indices
        vis_idx = visible.nonzero(as_tuple=False)[:, 1].reshape(B, num_visible)
        tgt_idx = mask.nonzero(as_tuple=False)[:, 1].reshape(B, num_target)

        # Context encoder: encode visible patches only
        context_emb = self.context_encoder.forward_features_masked(x, mask)

        # Target encoder: encode ALL patches (no grad)
        with torch.no_grad():
            target_emb = self.target_encoder.forward_features(x)  # (B, 49, D)

        # Extract target embeddings at masked positions
        D = target_emb.shape[-1]
        tgt_exp = tgt_idx.unsqueeze(-1).expand(-1, -1, D)
        target_at_masked = torch.gather(target_emb, 1, tgt_exp)  # (B, T, D)

        # Predictor: predict target representations
        predicted = self.predictor(context_emb, vis_idx, tgt_idx)  # (B, T, D)

        # L2 loss
        loss = F.mse_loss(predicted, target_at_masked.detach())
        return loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_jepa(config=None):
    """Train JEPA encoder. Returns path to saved encoder checkpoint."""
    cfg = {**DEFAULT_JEPA_CONFIG, **(config or {})}

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = get_device()
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Model
    encoder = VisionTransformer(
        in_channels=cfg["in_channels"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        feature_dim=cfg["feature_dim"],
    )
    predictor = JEPAPredictor(
        num_patches=encoder.num_patches,
        encoder_embed_dim=cfg["embed_dim"],
        predictor_embed_dim=cfg["predictor_embed_dim"],
        num_heads=cfg["predictor_num_heads"],
        num_layers=cfg["predictor_num_layers"],
    )
    model = JEPA(encoder, predictor, ema_decay=cfg["ema_decay"]).to(device)

    # Data
    dataset = FrameDataset(cfg["frames_path"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        shuffle=True, num_workers=0, drop_last=True)

    # Optimizer + scheduler
    params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] * len(loader)
    )

    # Logging
    writer = SummaryWriter(str(save_dir / "tb"))
    grid_size = int(encoder.num_patches ** 0.5)
    global_step = 0
    start_time = time.time()

    print(f"Training JEPA: {len(dataset)} frames, {cfg['epochs']} epochs")
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")
    print(f"Device: {device}")

    for epoch in range(1, cfg["epochs"] + 1):
        epoch_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = batch.to(device)
            mask = generate_block_mask(
                batch.shape[0], grid_size=grid_size,
                mask_ratio=cfg["mask_ratio"], device=device,
            )

            loss = model(batch, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            model.update_target_encoder()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            writer.add_scalar("loss/jepa", loss.item(), global_step)

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        writer.add_scalar("loss/epoch_avg", avg_loss, epoch)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch}/{cfg['epochs']} | loss {avg_loss:.6f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

        if epoch % 10 == 0:
            torch.save({
                "encoder_state_dict": model.context_encoder.state_dict(),
                "config": cfg,
                "epoch": epoch,
            }, save_dir / f"checkpoint_epoch{epoch}.pt")

    # Final save
    final_path = save_dir / "encoder_final.pt"
    torch.save({
        "encoder_state_dict": model.context_encoder.state_dict(),
        "config": cfg,
        "epoch": cfg["epochs"],
    }, final_path)
    print(f"JEPA training complete. Encoder saved to {final_path}")

    writer.close()
    return str(final_path)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jepa_encoder(checkpoint_path, device="cpu"):
    """Load a trained JEPA encoder. Returns a VisionTransformer."""
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
    return encoder.to(device)
