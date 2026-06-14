#!/usr/bin/env bash
# Gate watcher for the PLAIN CliffordCLIP arm. Polls the training log and exits
# (re-invoking the orchestrator) when the CLIP stage reaches ~epoch 15 (gate
# point), the training process dies, or a safety cap elapses. Transient file.
REPO=/media/arxwn/data_fast/repositories/dl_techniques
LOG=$(cat "$REPO/logs/.plain_current_log")
PID=$(cat "$REPO/logs/.plain_current_pid")
WLOG="$REPO/logs/.plain_watcher.log"
START=$(date +%s)
CAP=$((10 * 3600))   # 10h safety cap (pretrain ~1.5h + 15 CLIP epochs ~5.6h ~= 7.1h)
echo "[watcher start $(date)] PID=$PID LOG=$LOG CAP=${CAP}s" > "$WLOG"
while true; do
  NOW=$(date +%s); EL=$((NOW - START))
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "RESULT=TRAINING_DIED elapsed=${EL}s" | tee -a "$WLOG"; exit 0
  fi
  if grep -qE "Epoch (1[5-9]|[2-4][0-9]|50)/50" "$LOG" 2>/dev/null; then
    echo "RESULT=GATE_READY elapsed=${EL}s (CLIP epoch>=15 reached)" | tee -a "$WLOG"; exit 0
  fi
  if [ "$EL" -gt "$CAP" ]; then
    echo "RESULT=WATCHER_TIMEOUT elapsed=${EL}s" | tee -a "$WLOG"; exit 0
  fi
  EPOCH=$(grep -oE "Epoch [0-9]+/50" "$LOG" 2>/dev/null | tail -1)
  PROBE=$(grep "Probe \[step" "$LOG" 2>/dev/null | tail -1)
  echo "[hb $(date +%H:%M:%S) el=${EL}s] stage=${EPOCH:-pretrain-or-load} | ${PROBE:-no-clip-probe-yet}" >> "$WLOG"
  sleep 600
done
