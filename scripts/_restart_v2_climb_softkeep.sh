#!/usr/bin/env bash
set -uo pipefail
cd "/home/david/Projekte/AI Projects/Ascent Player"

kill_pat() {
  local pat="$1"
  local sig="$2"
  local pids
  pids=$(pgrep -f "$pat" || true)
  if [[ -n "${pids}" ]]; then
    echo "killing ($sig): $pat -> $pids"
    # shellcheck disable=SC2086
    kill "-$sig" $pids 2>/dev/null || true
  fi
}

kill_pat 'run_v2_climb_with_watchdog.py' TERM
kill_pat 'run_v2_climb_ladder.py' TERM
sleep 3
kill_pat 'run_v2_climb_with_watchdog.py' KILL
kill_pat 'run_v2_climb_ladder.py' KILL
pkill -9 -f 'chrome-linux64/chrome --disable-field-trial' 2>/dev/null || true
sleep 2

# Preserve v2_best; restore work checkpoint from V2_BEST.
if [[ -f checkpoints/dqn_v2_best.keras ]]; then
  cp -f checkpoints/dqn_v2_best.keras checkpoints/dqn_thread_bc.keras
  cp -f checkpoints/dqn_v2_best.weights.h5 checkpoints/dqn_thread_bc.weights.h5 2>/dev/null || true
  cp -f checkpoints/dqn_v2_best.meta.json checkpoints/dqn_thread_bc.meta.json 2>/dev/null || true
  cp -f checkpoints/dqn_v2_best.keras checkpoints/dqn_latest.keras
  echo "restored THREAD_BC/LATEST from V2_BEST"
fi
# Do not reset v2_probe_best_mean.txt

STAMP=$(date +%Y%m%d_%H%M%S)
WLOG="logs/v2_climb_watchdog_supervisor_${STAMP}.log"
echo "$WLOG" > logs/v2_climb_watchdog_supervisor.path

# Remaining wall ~7h from original 8h @ 22:00
nohup env PYTHONPATH=. PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/run_v2_climb_with_watchdog.py \
  --hours 7 \
  --run-seed 424242 \
  --target-mean 2000 \
  --stall-seconds 420 \
  --check-every 25 \
  --bc-steps 200 \
  --bc-lr 5e-7 \
  --td-steps 400 \
  --td-lr 5e-7 \
  --collect-minutes 6 \
  --probe-episodes 8 \
  --bootstrap-keep-floor 120 \
  --reliability-episodes 50 \
  --reliability-every 8 \
  > "$WLOG" 2>&1 &
echo $! > logs/v2_climb_watchdog.pid

kill_pat 'scripts/training_monitor.py' TERM
sleep 1
nohup env PYTHONPATH=. DISPLAY="${DISPLAY:-:0}" .venv/bin/python -u scripts/training_monitor.py \
  > logs/training_monitor_ui.log 2>&1 &
echo $! > logs/training_monitor_ui.pid

sleep 10
echo "RESTARTED stamp=$STAMP"
echo "watchdog_pid=$(cat logs/v2_climb_watchdog.pid)"
echo "monitor_pid=$(cat logs/training_monitor_ui.pid)"
echo "supervisor=$WLOG"
head -25 "$WLOG" || true
echo '--- procs ---'
ps -ef | grep -E '[.]venv/bin/python.*(run_v2_climb|training_monitor)' | grep -v grep || true
echo '--- latest ---'
cat logs/v2_climb_ladder_latest.path 2>/dev/null || echo "latest.path pending"
echo -n 'v2_best='; cat logs/v2_probe_best_mean.txt
