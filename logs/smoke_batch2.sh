#!/usr/bin/env bash
# Final smoke batch: validate steps 6/7/9 viz + runtime. Serial, GPU1.
# mdn: also validates SC-7 (--result_dir honored + grid from --plot_top_k_patterns 4).
set -u
cd /media/arxwn/data_fast/repositories/dl_techniques
export MPLBACKEND=Agg PYTHONPATH=src
SUMMARY=logs/smoke_batch2_summary.txt
: > "$SUMMARY"

run_one () {
  local m="$1" rd="$2" extra="$3" tmo="$4"
  local LOG="logs/smoke2_$m.log"
  echo "================ $m START (rd=$rd) ================" | tee -a "$SUMMARY"
  timeout "$tmo" .venv/bin/python -m train.time_series.$m.train_$m \
    --epochs 3 --steps_per_epoch 20 --n_samples 2000 --visualize_every_n_epochs 1 \
    --gpu 1 --result_dir "$rd" $extra > "$LOG" 2>&1
  local EXIT=$?
  local COMPLETED RESDIR RJSON PNG NEST UNDER
  COMPLETED=$(grep -c "Completed. Results:" "$LOG")
  RESDIR=$(grep -oP 'Completed\. Results:\s*\K\S+' "$LOG" | tail -1)
  RJSON="MISSING"; PNG=0; NEST="ok"; UNDER="no"
  if [ -n "$RESDIR" ] && [ -d "$RESDIR" ]; then
    [ -f "$RESDIR/results.json" ] && RJSON="present"
    PNG=$(find "$RESDIR/visualizations" -name '*.png' 2>/dev/null | wc -l)
    find "$RESDIR" -type d -name results 2>/dev/null | grep -q . && NEST="DOUBLE-NEST"
    case "$RESDIR" in "$rd"/*) UNDER="yes";; esac
  fi
  local V="FAIL"
  [ "$COMPLETED" -ge 1 ] && [ "$RJSON" = "present" ] && [ "$PNG" -ge 1 ] && [ "$NEST" = "ok" ] && [ "$UNDER" = "yes" ] && V="PASS"
  echo "$m: VERDICT=$V exit=$EXIT completed=$COMPLETED results.json=$RJSON png=$PNG nest=$NEST under_result_dir=$UNDER dir=$RESDIR" | tee -a "$SUMMARY"
  [ "$V" = "FAIL" ] && { echo "---- tail $LOG ----" | tee -a "$SUMMARY"; tail -12 "$LOG" | tee -a "$SUMMARY"; }
  echo "================ $m END ================" | tee -a "$SUMMARY"
}

run_one tirex   results/smoke2_tirex   ""                        420
run_one prism   results/smoke2_prism   ""                        420
run_one mdn     results/smoke2_mdn     "--plot_top_k_patterns 4"  420
run_one deepar  results/smoke2_deepar  ""                        900

echo "" | tee -a "$SUMMARY"; echo "==== BATCH2 COMPLETE ====" | tee -a "$SUMMARY"
grep "VERDICT=" "$SUMMARY"
