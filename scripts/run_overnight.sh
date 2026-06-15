#!/usr/bin/env bash
# Launch unsupervised overnight browser training (9-hour budget).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG="logs/overnight/console.log"
mkdir -p logs/overnight
echo "Starting overnight training at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
exec python -u -m ascent_player.overnight_browser "$@" 2>&1 | tee -a "$LOG"
