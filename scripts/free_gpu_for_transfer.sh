#!/usr/bin/env bash
# Best-effort free VRAM before browser transfer (Ollama often holds ~5GB).
# Usage: ./scripts/free_gpu_for_transfer.sh
set -euo pipefail
echo "GPU before:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
if pgrep -f 'ollama/llama-server' >/dev/null 2>&1; then
  echo "Stopping ollama llama-server to free VRAM..."
  pkill -f 'ollama/llama-server' || true
  sleep 2
fi
echo "GPU after:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader || true
