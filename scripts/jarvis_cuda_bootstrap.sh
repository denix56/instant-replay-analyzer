#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/home/instant-replay-analyzer}"
cd "$PROJECT_DIR"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "CUDA GPU is not visible in this Jarvis instance." >&2
  exit 2
fi

uv python install 3.13
rm -rf .venv
uv venv --python 3.13 --seed .venv
source .venv/bin/activate

python -m pip install --upgrade "pip>=25" "setuptools<82" wheel
uv pip install -e .

# Install attention/runtime extras only after Torch and CUDA runtime packages are present.
uv pip install "flash-linear-attention>=0.5.0,<1.0" "causal-conv1d>=1.6.2.post1,<2.0"

python - <<'PY'
import sys
import torch

print("python", sys.version)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cuda_version", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda is not available")
print("gpu", torch.cuda.get_device_name(0))

import causal_conv1d  # noqa: F401
import fla  # noqa: F401

print("causal_conv1d ok")
print("flash_linear_attention ok")
PY
