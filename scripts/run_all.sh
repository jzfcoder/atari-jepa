#!/usr/bin/env bash
# Full training pipeline: V0 pretraining + V1 experiments.
#
# Designed to run unattended on a GPU instance. Total wall time on an A10:
#   V0 pretraining (frames + JEPA + AE):  ~1.5 hours
#   V1 training (5 conditions x 3 seeds): ~15 hours
#   V1 evaluation:                         ~2 hours
#
# Usage:
#   bash scripts/run_all.sh              # run everything
#   bash scripts/run_all.sh --skip-v0    # skip V0 pretraining (encoders must exist)
#   bash scripts/run_all.sh --v1-only jepa_finetune vit_scratch  # run subset of V1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# -----------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------
SKIP_V0=false
V1_CONDITIONS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-v0)
            SKIP_V0=true
            shift
            ;;
        --v1-only)
            shift
            V1_CONDITIONS="--conditions $*"
            break
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash scripts/run_all.sh [--skip-v0] [--v1-only cond1 cond2 ...]"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
log() { echo -e "\n$(date '+%H:%M:%S') === $1 ===\n"; }

# -----------------------------------------------------------------------
# V0: Pretraining (collect frames, train JEPA, train AE)
# -----------------------------------------------------------------------
if [ "$SKIP_V0" = false ]; then
    log "V0 Step 1/3: Collecting gameplay frames"
    uv run python scripts/collect_frames.py

    log "V0 Step 2/3: Training JEPA encoder"
    uv run python scripts/train_jepa.py --config configs/jepa.yaml

    log "V0 Step 3/3: Training autoencoder encoder"
    uv run python scripts/train_autoencoder.py --config configs/autoencoder.yaml
else
    log "Skipping V0 pretraining (--skip-v0)"
    # Verify encoders exist
    if [ ! -f results/v0/jepa/encoder_final.pt ]; then
        echo "ERROR: results/v0/jepa/encoder_final.pt not found. Run without --skip-v0."
        exit 1
    fi
    if [ ! -f results/v0/autoencoder/encoder_final.pt ]; then
        echo "ERROR: results/v0/autoencoder/encoder_final.pt not found. Run without --skip-v0."
        exit 1
    fi
fi

# -----------------------------------------------------------------------
# V1: Training + Evaluation
# -----------------------------------------------------------------------
log "V1: Training and evaluating all conditions"
# shellcheck disable=SC2086
uv run python scripts/run_v1.py --config configs/v1.yaml $V1_CONDITIONS

log "Done! Results in results/v1/"
