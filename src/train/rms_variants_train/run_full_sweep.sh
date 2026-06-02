#!/usr/bin/env bash
# Full Phase 3 sweep — chained, exit on first failure.
# Generated from PHASE3_PLAN.md v3 (plan_2026-05-18_6776f8ba).
# Total estimated wall-clock: 50-70h on GPU 0 (RTX 4090).
# Serial GPU only (HC1). NEVER run in parallel with another GPU job.
set -euo pipefail

REPO=/media/arxwn/data_fast/repositories/dl_techniques
cd "$REPO"
mkdir -p logs results/rms_variants_train/p3_full

TS="$(date +%F_%H%M%S)"
PY="$REPO/.venv/bin/python"
NORMS_8="rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm,adaptive_band_rms,band_logit_norm,dynamic_tanh,zero_centered_adaptive_band_rms_norm"
NORMS_4="rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm"

export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=2

log() { printf "[%s] %s\n" "$(date +%F_%H:%M:%S)" "$*"; }

run_sweep() {
  local name="$1"; shift
  log "=== START $name ==="
  cd "$REPO/src"
  "$PY" -m train.rms_variants_train.sweep "$@" 2>&1 | tee "$REPO/logs/p3_${name}.${TS}.log"
  cd "$REPO"
  log "=== END   $name ==="
}

# ---------------- 0. Smoke gate SKIPPED ----------------
# Option 1 per user 2026-05-18: smoke caps in PHASE3_PLAN.md v3 were too tight
# (90s/cell × 300s global insufficient for TF subprocess cold-start). Harness
# integrity already proven by 637/637 unit tests + test_e6_clm.py 1-step CPU smoke.
log "=== smoke gate SKIPPED (option 1) ==="

# ---------------- 1. Chunk 1 — Core OOB ----------------
run_sweep chunk1 \
  --gpu 0 \
  --experiments e1,e2,e3,e4,e5 \
  --norms "$NORMS_8" \
  --modes oob --regimes default --seeds 0,1,2,3,4 \
  --max-cells 250 --global-cap-s 64800 \
  --out-dir "$REPO/results/rms_variants_train/p3_chunk1"

# ---------------- 2. Chunk 2 — param_matched ----------------
run_sweep chunk2 \
  --gpu 0 \
  --experiments e1,e2,e3,e4,e5 \
  --norms "$NORMS_4" \
  --modes param_matched --regimes default --seeds 0,1,2,3,4 \
  --max-cells 150 --global-cap-s 32400 \
  --out-dir "$REPO/results/rms_variants_train/p3_chunk2"

# ---------------- 3a. Chunk 3 — E5 regimes ----------------
run_sweep chunk3_e5 \
  --gpu 0 --experiments e5 \
  --norms "$NORMS_8" \
  --modes oob \
  --regimes default,bs_32,bs_256,lr_low,lr_high,lr_extreme,wd_zero \
  --seeds 0,1,2,3,4 \
  --max-cells 300 --global-cap-s 21600 \
  --out-dir "$REPO/results/rms_variants_train/p3_chunk3_e5"

# ---------------- 3b. Chunk 3 — E1 regimes ----------------
run_sweep chunk3_e1 \
  --gpu 0 --experiments e1 \
  --norms "$NORMS_8" \
  --modes oob \
  --regimes default,lr_low,lr_high,mp_fp16,lr_extreme,wd_zero,bs_4,mp_fp16_lowloss \
  --seeds 0,1,2,3,4 \
  --max-cells 350 --global-cap-s 43200 \
  --out-dir "$REPO/results/rms_variants_train/p3_chunk3_e1"

# ---------------- 4. Chunk 4 — E6 causal-LM ----------------
run_sweep chunk4_e6 \
  --gpu 0 --experiments e6 \
  --norms "$NORMS_8" \
  --modes oob --regimes default --seeds 0,1,2 \
  --max-cells 50 --global-cap-s 14400 \
  --out-dir "$REPO/results/rms_variants_train/p3_chunk4_e6"

# ---------------- 5. Overhead bench ----------------
log "=== START overhead ==="
cd "$REPO/src"
"$PY" -m train.rms_variants_train.norm_overhead_bench \
  --out "$REPO/results/rms_variants_train/p3_full/overhead.csv" \
  2>&1 | tee "$REPO/logs/p3_overhead.${TS}.log"
cd "$REPO"
log "=== END   overhead ==="

# ---------------- 6. Aggregate + report ----------------
log "=== START aggregate+report ==="
"$PY" - <<'PYEOF'
import pandas as pd, glob
frames = [pd.read_csv(p) for p in sorted(glob.glob('results/rms_variants_train/p3_chunk*/aggregated.csv'))]
pd.concat(frames, ignore_index=True).to_csv(
    'results/rms_variants_train/p3_full/aggregated.csv', index=False)
print(f'Aggregated {len(frames)} chunk CSVs.')
PYEOF

cd "$REPO/src"
"$PY" -m train.rms_variants_train.report \
  --in-dir "$REPO/results/rms_variants_train/p3_full" \
  2>&1 | tee "$REPO/logs/p3_report.${TS}.log"
cd "$REPO"
log "=== END   aggregate+report ==="

log "ALL DONE. See results/rms_variants_train/p3_full/summary.md"
