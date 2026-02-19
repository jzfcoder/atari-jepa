# JEPA Encoders for Visually Robust Atari Agents

Can self-supervised [JEPA](https://arxiv.org/abs/2301.08243) pretraining make
RL agents more robust to visual perturbations? And can fine-tuning pretrained
encoders close the clean-performance gap while preserving that robustness?

This project investigates these questions across two experiment rounds on
Breakout, comparing CNN, JEPA, and autoencoder encoders under visual corruption.

<p align="center">
  <img src="results/v0/gifs/comparison_clean.gif" width="300" alt="Agent playing clean Breakout">
  <img src="results/v0/gifs/comparison_hard.gif" width="300" alt="Agent under hard perturbations">
</p>
<p align="center"><em>Left: clean environment. Right: hard visual perturbations (same agent).</em></p>

## Key Findings

1. **JEPA pretraining produces more robust representations** — frozen JEPA
   encoders retain more performance under visual corruption than end-to-end CNNs
2. **Robustness comes from the pretraining objective, not the architecture** —
   a randomly initialized ViT (same architecture, no pretraining) collapses
   under perturbation
3. **Fine-tuning preserves robustness** — JEPA fine-tune improves clean
   performance by 47% over frozen JEPA while maintaining nearly identical
   robustness ratio
4. **CNNs remain more practical for Atari RL** — the ViT architecture's sample
   inefficiency creates a large clean-performance gap that fine-tuning cannot
   fully close at practical training budgets

## V0: Frozen Encoder Experiments

**Question**: Does a frozen JEPA encoder make a PPO agent more robust?

Trained a JEPA encoder (ViT-Tiny) and an autoencoder (same architecture) via
self-supervised learning on Breakout gameplay frames, then froze each encoder
and trained PPO policy heads on top.

| Condition | Clean | Hard | Robustness Ratio |
|---|---|---|---|
| Stock CNN | **7.6** +/- 0.3 | 2.2 +/- 0.1 | 0.612 |
| JEPA (frozen) | 5.6 +/- 0.4 | **3.1** +/- 0.9 | **0.764** |
| Autoencoder | 2.0 +/- 0.1 | 2.0 +/- 0.0 | 0.971* |

*\*Autoencoder never exceeded random-policy performance.*

**Result**: JEPA retains 76% of clean performance under perturbation vs 61% for
CNN. Under hard perturbations, JEPA (3.1) outperforms the CNN (2.2) in absolute
terms despite lower clean scores. The autoencoder fails entirely, showing that
pixel-reconstruction pretraining doesn't produce RL-useful representations.

<p align="center">
  <img src="results/v0/phase4/robustness_bar.png" width="480" alt="V0 robustness bar chart">
  <img src="results/v0/phase4/robustness_ratio.png" width="400" alt="V0 robustness ratio">
</p>

Full details: [V0 Report](results/v0/phase4/REPORT.md)

## V1: Fine-Tuning Experiments

**Question**: Can fine-tuning JEPA encoders during RL close the clean gap while
preserving robustness?

Added three new conditions: JEPA fine-tune (0.1x encoder LR), AE fine-tune
(0.1x encoder LR), and ViT scratch (random init, full LR as architecture
control). 15 runs total (5 conditions x 3 seeds).

| Condition | Clean | Hard | Robustness Ratio |
|---|---|---|---|
| Stock CNN | **8.1** +/- 0.8 | **2.7** +/- 0.7 | 0.617 |
| JEPA (frozen) | 3.6 +/- 1.9 | 2.3 +/- 0.0 | **0.659** |
| JEPA (fine-tune) | 5.3 +/- 1.1 | 2.2 +/- 0.5 | 0.656 |
| AE (fine-tune) | 4.7 +/- 1.0 | 1.9 +/- 0.1 | 0.631 |
| ViT (scratch) | 4.9 +/- 0.6 | 0.6 +/- 0.7 | 0.266 |

**Result**: Fine-tuning improves JEPA's clean reward by 47% (3.6 → 5.3) with
no robustness loss (0.659 → 0.656 ratio). ViT scratch collapses under
perturbation (0.266 ratio), proving robustness comes from JEPA pretraining, not
the ViT architecture. However, Stock CNN still leads clean performance (8.1) and
learning curves show it hasn't converged — the ViT architecture is fundamentally
less sample-efficient for 84x84 Atari RL.

<p align="center">
  <img src="results/v1/plots/v1_pareto.png" width="420" alt="V1 Pareto">
  <img src="results/v1/plots/v1_robustness_ratio.png" width="480" alt="V1 robustness ratio">
</p>

Full details: [V1 Results](V1_RESULTS.md)

## Motivation

RL agents trained on pixel observations are brittle: small visual changes
(color shifts, noise, brightness) can destroy performance even when the
underlying game state is identical. This mirrors the sim-to-real gap where
agents trained in simulation fail under real-world visual variation.

JEPA (Joint Embedding Predictive Architecture) learns to predict abstract
*representations* of masked image patches rather than reconstructing raw pixels.
The hypothesis is that this forces the encoder to capture scene structure (ball
position, paddle location) rather than surface-level pixel details, producing
representations that are inherently more robust to visual perturbation.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jzfcoder/atari-jepa.git
cd atari-jepa
uv sync
```

Hardware: runs on Mac (MPS), CUDA GPUs, or CPU.

## Usage

### V0: Frozen encoder experiments

```bash
# Phase 1: Train baseline PPO agent
uv run python scripts/train_baseline.py

# Phase 2: Self-supervised encoder pretraining
uv run python scripts/collect_frames.py
uv run python scripts/train_jepa.py
uv run python scripts/train_autoencoder.py

# Phase 3: Encoder-swap experiments (3 seeds x 3 conditions)
uv run python scripts/run_phase3.py

# Phase 4: Analysis
uv run python scripts/utils/plot_robustness.py
uv run python scripts/utils/plot_learning_curves.py
```

### V1: Fine-tuning experiments

```bash
# Train all 15 runs (requires V0 pretrained encoders)
uv run python scripts/run_v1.py

# Evaluation only
uv run python scripts/run_v1.py --skip-training

# Generate plots
uv run python scripts/plot_v1.py
uv run python scripts/plot_learning_curves_v1.py
```

See [TRAINING.md](TRAINING.md) for GPU setup and Lambda instructions.

## Architecture

```
agents/
  ppo_atari.py       # CleanRL-style PPO with swappable encoder + fine-tune mode
  encoder.py         # ViT-Tiny (patch_size=12, embed_dim=192, 4 layers, ~2M params)
  jepa.py            # JEPA: masked patch prediction in representation space
  autoencoder.py     # Pixel-reconstruction baseline (same ViT architecture)
env/
  perturbations.py   # Color jitter, noise, combined perturbation wrappers
  wrappers.py        # Standard Atari preprocessing
scripts/             # Training, evaluation, and analysis scripts
configs/             # YAML configs for each experiment phase
```

`ppo_atari.Agent` takes an optional `encoder` argument. When provided, the
encoder can be frozen (V0-style) or fine-tuned with a separate learning rate
(V1-style). This makes it trivial to swap between CNN, JEPA, and autoencoder
encoders while keeping everything else identical.

## License

[MIT](LICENSE)
