#!/usr/bin/env bash
# Launch Ascent Player with NVIDIA pip libraries visible to TensorFlow.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
PYTHON="${VENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing venv at ${VENV}. Create it with: python3.11 -m venv .venv && pip install -r requirements.txt" >&2
  exit 1
fi
if NVIDIA_LIBS="$(find "${VENV}/lib" -path '*/site-packages/nvidia/*/lib' -type d 2>/dev/null | tr '\n' ':')"; then
  export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
fi
exec "${PYTHON}" "${ROOT}/main.py" "$@"
