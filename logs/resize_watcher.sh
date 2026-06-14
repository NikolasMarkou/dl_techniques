#!/usr/bin/env bash
# Watches the CC3M->SSD resize job. Exits (re-invoking the orchestrator) when
# the job logs "ALL DONE", the process dies, or a safety cap elapses.
REPO=/media/arxwn/data_fast/repositories/dl_techniques
LOG=$(cat "$REPO/logs/.resize_log")
PID=$(cat "$REPO/logs/.resize_pidf")
WLOG="$REPO/logs/.resize_watcher.log"
START=$(date +%s)
CAP=$((9 * 3600))
echo "[resize-watcher start $(date)] PID=$PID LOG=$LOG" > "$WLOG"
while true; do
  NOW=$(date +%s); EL=$((NOW - START))
  if grep -q "ALL DONE" "$LOG" 2>/dev/null; then
    echo "RESULT=RESIZE_DONE elapsed=${EL}s" | tee -a "$WLOG"
    grep -E "DONE|ALL DONE|Linked" "$LOG" | tail -6 | tee -a "$WLOG"; exit 0
  fi
  if ! pgrep -f "train.cliffordnet.resize_cc3m_to_ssd" >/dev/null 2>&1; then
    echo "RESULT=RESIZE_DIED elapsed=${EL}s (no ALL DONE marker, no resize process)" | tee -a "$WLOG"; exit 0
  fi
  if [ "$EL" -gt "$CAP" ]; then
    echo "RESULT=RESIZE_TIMEOUT elapsed=${EL}s" | tee -a "$WLOG"; exit 0
  fi
  HB=$(grep -E "img/s|\] DONE" "$LOG" 2>/dev/null | tail -1)
  echo "[hb $(date +%H:%M:%S) el=${EL}s] ${HB:-starting}" >> "$WLOG"
  sleep 900
done
