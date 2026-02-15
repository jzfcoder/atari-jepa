# Training Guide

How to reproduce all results from scratch on a GPU instance.

## Quick start (Lambda / any CUDA machine)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart shell

# 2. Clone and install
git clone https://github.com/jzfcoder/atari-jepa.git
cd atari-jepa
uv sync

# 3. Run everything (V0 pretraining + V1 experiments)
bash scripts/run_all.sh
```

Results land in `results/v1/`. Total wall time: ~18 hours on A10, ~8 hours on A100.

## What the pipeline does

### V0: Encoder pretraining

These produce the pretrained JEPA and autoencoder encoders that V1 builds on.

```bash
# 1. Collect ~100K gameplay frames using a random policy
uv run python scripts/collect_frames.py
# → results/v0/frames.npz

# 2. Train JEPA encoder (self-supervised, ~100 epochs)
uv run python scripts/train_jepa.py --config configs/jepa.yaml
# → results/v0/jepa/encoder_final.pt

# 3. Train autoencoder encoder (pixel reconstruction baseline)
uv run python scripts/train_autoencoder.py --config configs/autoencoder.yaml
# → results/v0/autoencoder/encoder_final.pt
```

### V1: Unfrozen encoder experiments

Five conditions, three seeds each (15 training runs + evaluation):

| Condition | Encoder | Init | Training Mode |
|---|---|---|---|
| `stock_cnn` | Nature-CNN | Random | End-to-end |
| `jepa_frozen` | ViT-Tiny | JEPA pretrained | Frozen encoder, train heads only |
| `jepa_finetune` | ViT-Tiny | JEPA pretrained | End-to-end (encoder LR × 0.1) |
| `ae_finetune` | ViT-Tiny | AE pretrained | End-to-end (encoder LR × 0.1) |
| `vit_scratch` | ViT-Tiny | Random | End-to-end (full LR) |

```bash
# Run all conditions (training + eval)
uv run python scripts/run_v1.py --config configs/v1.yaml

# Run a subset of conditions
uv run python scripts/run_v1.py --conditions jepa_finetune vit_scratch

# Evaluate only (models must already exist)
uv run python scripts/run_v1.py --skip-training
```

Results are saved to:
- `results/v1/v1_results.txt` — comparison table
- `results/v1/v1_results.json` — machine-readable results
- `results/v1/<condition>/<condition>_seed<N>/tb/` — TensorBoard logs

## Running on Lambda Labs

1. Launch a GPU instance (A10 is cost-effective, A100/H100 for speed)
2. SSH in and follow the quick start above
3. To run in background (survives SSH disconnect):
   ```bash
   nohup bash scripts/run_all.sh > train.log 2>&1 &
   tail -f train.log  # watch progress
   ```
4. Monitor with TensorBoard:
   ```bash
   uv run tensorboard --logdir results/v1 --bind_all --port 6006
   ```
   Then open `http://<instance-ip>:6006` in your browser.

## Running a subset

If you only care about specific conditions or want to iterate faster:

```bash
# Just the new fine-tuning conditions (skip stock_cnn and jepa_frozen)
uv run python scripts/run_v1.py --conditions jepa_finetune ae_finetune vit_scratch

# Quick smoke test (tiny run, ~30 seconds)
uv run python -c "
from agents.encoder import VisionTransformer
from agents.ppo_atari import train
enc = VisionTransformer()
train(config={
    'total_timesteps': 2048, 'num_envs': 2, 'num_steps': 64,
    'save_interval': 0, 'save_dir': '/tmp/smoke',
    'freeze_encoder': False, 'encoder_lr_scale': 0.1,
}, encoder=enc)
"
```

## Outputs

```
results/
  v0/
    frames.npz                     # gameplay frames for pretraining
    jepa/encoder_final.pt          # pretrained JEPA encoder
    autoencoder/encoder_final.pt   # pretrained AE encoder
  v1/
    stock_cnn/stock_cnn_seed1/     # trained models + TB logs
    jepa_frozen/jepa_frozen_seed1/
    jepa_finetune/...
    ae_finetune/...
    vit_scratch/...
    v1_results.txt                 # comparison table
    v1_results.json                # full results
```
