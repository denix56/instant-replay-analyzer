#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

BOOTSTRAP="${BOOTSTRAP:-auto}"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

needs_bootstrap=0
if [[ "$BOOTSTRAP" == "1" || "$BOOTSTRAP" == "true" ]]; then
  needs_bootstrap=1
elif [[ "$BOOTSTRAP" == "auto" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    needs_bootstrap=1
  elif ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
import torch
import causal_conv1d  # noqa: F401
import fla  # noqa: F401

if sys.version_info[:2] != (3, 13):
    raise SystemExit(1)
if not torch.cuda.is_available():
    raise SystemExit(1)
PY
  then
    needs_bootstrap=1
  fi
fi

if [[ "$needs_bootstrap" == "1" ]]; then
  "$PROJECT_DIR/scripts/jarvis_cuda_bootstrap.sh" "$PROJECT_DIR"
fi

source "$PROJECT_DIR/.venv/bin/activate"
if [[ -f "$PROJECT_DIR/.env.remote" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env.remote"
  set +a
fi

export PYTHONPATH="${PYTHONPATH:-backend}"
export GPU_BACKEND="${GPU_BACKEND:-cuda}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python "$PROJECT_DIR/scripts/jarvis_summary_smoke.py" "$@"
