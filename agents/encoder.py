"""ViT-Tiny encoder for 84x84 Atari frames (~1M params).

Designed for use in JEPA and autoencoder pre-training, and as a drop-in
replacement for the CNN encoder in the PPO Agent via the standard interface:
    - Class attribute FEATURE_DIM (int)
    - forward(x) : (batch, 4, 84, 84) -> (batch, FEATURE_DIM)

Also exposes forward_features() and forward_features_masked() for
self-supervised training (JEPA predictor, AE decoder, etc.).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trunc_normal_(tensor, std=0.02):
    """Truncated normal initialization (values beyond 2*std are redrawn)."""
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Tokenize an image into patch embeddings via Conv2d.

    For 84x84 with patch_size=12: 7x7 = 49 patches.
    """

    def __init__(self, in_channels=4, embed_dim=192, patch_size=12, img_size=84):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                       # (B, D, G, G)
        return x.flatten(2).transpose(1, 2)    # (B, num_patches, D)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer: LN -> MHSA -> res -> LN -> MLP -> res."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Vision Transformer
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """ViT-Tiny for 84x84 Atari frames.

    Three forward paths:
        forward(x)                     -> (B, feature_dim)            pooled CLS
        forward_features(x)           -> (B, num_patches, embed_dim)  per-patch
        forward_features_masked(x, m) -> (B, num_visible, embed_dim)  visible only
    """

    FEATURE_DIM = 512

    def __init__(self, in_channels=4, patch_size=12, embed_dim=192,
                 num_heads=3, num_layers=4, feature_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.FEATURE_DIM = feature_dim

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.num_patches = self.patch_embed.num_patches

        # Positional embeddings: index 0 = CLS, 1..N = patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, feature_dim)

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode(self, x):
        """Shared encoding: patch embed + CLS + pos + transformer + norm."""
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)

    def forward(self, x):
        """(B, 4, 84, 84) -> (B, feature_dim)."""
        tokens = self._encode(x)
        return self.head(tokens[:, 0])

    def forward_features(self, x):
        """(B, 4, 84, 84) -> (B, num_patches, embed_dim)."""
        tokens = self._encode(x)
        return tokens[:, 1:]

    def forward_features_masked(self, x, mask):
        """Encode only visible (unmasked) patches.

        Args:
            x:    (B, 4, 84, 84) float32.
            mask: (B, num_patches) bool, True = MASKED (not visible).

        Returns:
            (B, num_visible, embed_dim).
        """
        B = x.shape[0]
        all_patches = self.patch_embed(x)       # (B, 49, D)
        visible = ~mask                          # True = visible

        num_visible = int(visible[0].sum().item())

        # Gather visible patch indices (assumes uniform count across batch)
        vis_idx = visible.nonzero(as_tuple=False)[:, 1].reshape(B, num_visible)
        idx_exp = vis_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)

        vis_patches = torch.gather(all_patches, 1, idx_exp)

        # Position embeddings for visible patches (offset by 1 for CLS)
        patch_pos = self.pos_embed[:, 1:].expand(B, -1, -1)
        vis_patches = vis_patches + torch.gather(patch_pos, 1, idx_exp)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1]
        tokens = torch.cat([cls, vis_patches], dim=1)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        return tokens[:, 1:]
