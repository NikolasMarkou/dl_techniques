#!/usr/bin/env bash
# Iter-2 STRETCH downsampling campaign — 13 runs serial on GPU 0.
# Launched autonomously. Each run takes ~115 min on RTX 4090.
set -u  # don't -e: keep going even if one variant fails
cd "$(dirname "$0")"

LOG_ROOT="results/iter2_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
MASTER_LOG="$LOG_ROOT/master.log"

echo "=== iter-2 sweep starting at $(date) ===" | tee -a "$MASTER_LOG"
echo "Log root: $LOG_ROOT" | tee -a "$MASTER_LOG"

run_one() {
  local variant="$1" seed="$2"
  local tag="${variant}_seed${seed}"
  local log="$LOG_ROOT/${tag}.log"
  echo "" | tee -a "$MASTER_LOG"
  echo "--- [$(date +%H:%M:%S)] START $tag ---" | tee -a "$MASTER_LOG"
  MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_experiments \
      --variant "$variant" --seed "$seed" \
      --epochs 100 --batch-size 128 --gpu 0 \
      >"$log" 2>&1
  local rc=$?
  echo "--- [$(date +%H:%M:%S)] END   $tag (rc=$rc) ---" | tee -a "$MASTER_LOG"
}

# A0: anchor seed panel — 9 runs
for V in V0_baseline_avg_avg V1_blur_blur V7_blur_pxsh_int_abs; do
  for S in 42 137 2025; do
    run_one "$V" "$S"
  done
done

# B + C: 4 follow-up cells, seed 42
for V in B1_blur_blur_abs B2_blur_blur_pyrdiff C1_blur_blur_gn_late C2_blur_blur_ln_late; do
  run_one "$V" 42
done

echo "" | tee -a "$MASTER_LOG"
echo "=== iter-2 sweep finished at $(date) ===" | tee -a "$MASTER_LOG"

# Aggregate all per-run comparison.csv files written during the sweep.
.venv/bin/python tools/aggregate_iter2.py "$LOG_ROOT" >>"$MASTER_LOG" 2>&1 || true
echo "Done. See $LOG_ROOT/master.log and $LOG_ROOT/aggregated.csv"
