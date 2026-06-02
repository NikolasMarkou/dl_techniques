#!/usr/bin/env bash
# Phase 3 n=15 — Chunk 1 only, GPU 1 (RTX 4070, 12 GB).
# 600 cells: 8 norms x 5 experiments x 15 seeds. ~3 days wall-clock.
# NO `set -e` on the sweep step — an OOM/timeout exit must NOT skip overhead+report.
set -uo pipefail

REPO=/media/arxwn/data_fast/repositories/dl_techniques
cd "$REPO"
mkdir -p logs results/rms_variants_train/p3n15_chunk1

TS="$(date +%F_%H%M%S)"
PY="$REPO/.venv/bin/python"
NORMS_8="rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm,adaptive_band_rms,band_logit_norm,dynamic_tanh,zero_centered_adaptive_band_rms_norm"
OUT="$REPO/results/rms_variants_train/p3n15_chunk1"

export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=2

log() { printf "[%s] %s\n" "$(date +%F_%H:%M:%S)" "$*"; }

# ---------------- 1. Chunk 1 — Core OOB, n=15 ----------------
log "=== START n15_chunk1 ==="
cd "$REPO/src"
CUDA_VISIBLE_DEVICES=1 "$PY" -m train.rms_variants_train.sweep \
  --gpu 1 \
  --experiments e1,e2,e3,e4,e5 \
  --norms "$NORMS_8" \
  --modes oob --regimes default \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
  --max-cells 650 --global-cap-s 360000 \
  --no-report \
  --out-dir "$OUT" \
  2>&1 | tee "$REPO/logs/p3n15_chunk1.${TS}.log"
log "=== END n15_chunk1 (exit ${PIPESTATUS[0]}) ==="
cd "$REPO"

# ---------------- 2. Overhead bench ----------------
log "=== START overhead ==="
cd "$REPO/src"
CUDA_VISIBLE_DEVICES=1 "$PY" -m train.rms_variants_train.norm_overhead_bench \
  --out "$OUT/overhead.csv" \
  2>&1 | tee "$REPO/logs/p3n15_overhead.${TS}.log"
log "=== END overhead (exit ${PIPESTATUS[0]}) ==="
cd "$REPO"

# ---------------- 3. Report ----------------
log "=== START report ==="
cd "$REPO/src"
"$PY" -m train.rms_variants_train.report \
  --in-dir "$OUT" \
  2>&1 | tee "$REPO/logs/p3n15_report.${TS}.log"
log "=== END report (exit ${PIPESTATUS[0]}) ==="
cd "$REPO"

log "n15_chunk1 DONE. See $OUT/summary.md"
