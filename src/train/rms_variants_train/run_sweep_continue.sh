#!/usr/bin/env bash
# Continuation: chunk4_e6 + overhead + aggregate + report.
# Chunks 1/2/3 already complete. NO `set -e` — a non-zero chunk exit
# (e.g. global-timeout abort) must NOT skip the remaining steps.
set -uo pipefail

REPO=/media/arxwn/data_fast/repositories/dl_techniques
cd "$REPO"
mkdir -p logs results/rms_variants_train/p3_full

TS="$(date +%F_%H%M%S)"
PY="$REPO/.venv/bin/python"
NORMS_8="rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm,adaptive_band_rms,band_logit_norm,dynamic_tanh,zero_centered_adaptive_band_rms_norm"

export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=2

log() { printf "[%s] %s\n" "$(date +%F_%H:%M:%S)" "$*"; }

# ---------------- 4. Chunk 4 — E6 causal-LM ----------------
log "=== START chunk4_e6 ==="
cd "$REPO/src"
"$PY" -m train.rms_variants_train.sweep \
  --gpu 0 --experiments e6 \
  --norms "$NORMS_8" \
  --modes oob --regimes default --seeds 0,1,2 \
  --max-cells 50 --global-cap-s 14400 \
  --out-dir "$REPO/results/rms_variants_train/p3_chunk4_e6" \
  2>&1 | tee "$REPO/logs/p3_chunk4_e6.${TS}.log"
log "=== END chunk4_e6 (exit ${PIPESTATUS[0]}) ==="
cd "$REPO"

# ---------------- 5. Overhead bench ----------------
log "=== START overhead ==="
cd "$REPO/src"
"$PY" -m train.rms_variants_train.norm_overhead_bench \
  --out "$REPO/results/rms_variants_train/p3_full/overhead.csv" \
  2>&1 | tee "$REPO/logs/p3_overhead.${TS}.log"
log "=== END overhead (exit ${PIPESTATUS[0]}) ==="
cd "$REPO"

# ---------------- 6. Aggregate + report ----------------
log "=== START aggregate+report ==="
"$PY" - <<'PYEOF'
import pandas as pd, glob
paths = sorted(glob.glob('results/rms_variants_train/p3_chunk*/all_runs.csv'))
frames = [pd.read_csv(p) for p in paths if pd.read_csv(p).shape[0] > 0]
if frames:
    pd.concat(frames, ignore_index=True).to_csv(
        'results/rms_variants_train/p3_full/all_runs.csv', index=False)
    print(f'Aggregated {len(frames)} chunk all_runs.csv -> {sum(len(f) for f in frames)} rows')
else:
    print('WARNING: no chunk all_runs.csv found')
PYEOF

cd "$REPO/src"
"$PY" -m train.rms_variants_train.report \
  --in-dir "$REPO/results/rms_variants_train/p3_full" \
  2>&1 | tee "$REPO/logs/p3_report.${TS}.log"
log "=== END aggregate+report (exit ${PIPESTATUS[0]}) ==="
cd "$REPO"

log "CONTINUATION DONE. See results/rms_variants_train/p3_full/summary.md"
