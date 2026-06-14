#!/usr/bin/env bash
# Step-4 infra smoke sweep: 7 TS trainers, serial, GPU1. Validates iter-1 steps 1-3.
# Oracle per trainer: "Completed. Results: <dir>" log line + <dir>/results.json + <dir>/visualizations/*.png + no double-nest.
set -u
cd /media/arxwn/data_fast/repositories/dl_techniques
export MPLBACKEND=Agg
export PYTHONPATH=src
LOGDIR=logs
SUMMARY=$LOGDIR/smoke_summary.txt
: > "$SUMMARY"
TRAINERS="tirex prism deepar xlstm mdn nbeats adaptive_ema"
COMMON_ARGS="--epochs 3 --steps_per_epoch 20 --n_samples 2000 --visualize_every_n_epochs 1 --gpu 1"

for m in $TRAINERS; do
  LOG=$LOGDIR/smoke_$m.log
  echo "================ SMOKE $m START ================" | tee -a "$SUMMARY"
  timeout 420 .venv/bin/python -m train.time_series.$m.train_$m $COMMON_ARGS > "$LOG" 2>&1
  EXIT=$?
  # Oracle checks
  COMPLETED=$(grep -c "Completed. Results:" "$LOG")
  RESDIR=$(grep -oP 'Completed\. Results:\s*\K\S+' "$LOG" | tail -1)
  RJSON="MISSING"; PNG=0; NEST="ok"
  if [ -n "$RESDIR" ] && [ -d "$RESDIR" ]; then
    [ -f "$RESDIR/results.json" ] && RJSON="present"
    PNG=$(find "$RESDIR/visualizations" -name '*.png' 2>/dev/null | wc -l)
    # double-nest check: a nested results/ dir inside the exp dir
    find "$RESDIR" -type d -name results 2>/dev/null | grep -q . && NEST="DOUBLE-NEST"
  fi
  VERDICT="FAIL"
  if [ "$COMPLETED" -ge 1 ] && [ "$RJSON" = "present" ] && [ "$PNG" -ge 1 ] && [ "$NEST" = "ok" ]; then
    VERDICT="PASS"
  fi
  echo "$m: VERDICT=$VERDICT exit=$EXIT completed=$COMPLETED results.json=$RJSON png=$PNG nest=$NEST dir=$RESDIR" | tee -a "$SUMMARY"
  # tail of log on failure for quick diagnosis
  if [ "$VERDICT" = "FAIL" ]; then
    echo "---- last 15 lines of $LOG ----" | tee -a "$SUMMARY"
    tail -15 "$LOG" | tee -a "$SUMMARY"
  fi
  echo "================ SMOKE $m END (exit=$EXIT) ================" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "==== SWEEP COMPLETE ====" | tee -a "$SUMMARY"
grep -E "VERDICT=" "$SUMMARY" | grep -v "START\|END"
