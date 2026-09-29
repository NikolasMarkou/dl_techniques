#!/usr/bin/env bash
# Bias-free ConvUNeXt depth sweep (depth 1, 2, 3) — frozen depthwise Gabor stem.
# Stem: --gabor-filters-per-channel 22 (channels=3 -> 66-wide), projected via the
# default 1x1 conv down to --initial-filters 64, then frozen (--freeze-gabor-stem).
# Flat channel width across levels (--filter-multiplier 1), 3 blocks/level,
# Laplacian-pyramid path with 3 high-freq blocks/level, batchnorm block norm,
# ConvNeXt v1. Sequential only — NEVER run GPU jobs in parallel.
set -uo pipefail

REPO=/media/arxwn/data_fast/repositories/dl_techniques
cd "$REPO"
mkdir -p logs

GPU="${1:-1}"
PY="$REPO/.venv/bin/python"
TS="$(date +%F_%H%M%S)"

export MPLBACKEND=Agg

log() { printf "[%s] %s\n" "$(date +%F_%H:%M:%S)" "$*"; }

COMMON_ARGS=(
  --convnext-version v1
  --filter-multiplier 1
  --blocks-per-level 3
  --laplacian-pyramid
  --high-freq-blocks 3
  --gabor-filters-per-channel 22
  --channels 3
  --initial-filters 64
  --freeze-gabor-stem
  --block-normalization batchnorm
  --batch-size 16
  --epochs 100
  --curriculum-epochs 80
  --learning-rate 1e-3
  --warmup-epochs 10
  --weight-decay 0.004
  --gradient-clipping 1.0
  --sigma-max-start 0.025
  --sigma-max-end 0.25
  --gpu "$GPU"
)

for DEPTH in 1 2 3; do
  EXP_NAME="bfconvunext_gabor_depthwise_d${DEPTH}"
  log "=== START depth=${DEPTH} (${EXP_NAME}) ==="
  "$PY" -m train.bfunet.train_convunext_denoiser \
    --depth "$DEPTH" \
    --experiment-name "$EXP_NAME" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$REPO/logs/${EXP_NAME}.${TS}.log"
  log "=== END depth=${DEPTH} (exit ${PIPESTATUS[0]}) ==="
done

log "depth sweep DONE. See $REPO/results/ for run directories."
