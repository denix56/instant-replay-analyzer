# Native UI

The desktop UI uses Tauri with React/TypeScript.

Responsibilities:

- native folder picker
- `uv`, Python, and FFmpeg detection
- runtime profile detection
- `.env` generation for native backend settings
- native backend auto-start on app launch plus manual start/stop
- setup progress
- dashboard status
- groups and indexing actions
- search and result cards
- clip detail and analysis actions
- logs and diagnostics

The UI calls the backend API at `http://127.0.0.1:8000`. It does not process videos or run model inference itself.

When `Auto-start native backend` is enabled, app launch automatically:

- checks for `uv`
- installs `uv` when missing
- runs `uv sync --project <repo> --no-dev`
- on Linux/NVIDIA, installs `causal-conv1d` and `flash-linear-attention` with `uv pip install` after the base PyTorch/CUDA environment is synced
- writes `.env` from the selected clips folder, runtime profile, and model tier
- starts `uv run --project <repo> python -m app serve --host 127.0.0.1 --port 8000`
- polls the backend health endpoint until the API is ready

First-run setup writes native values:

- `CLIPS_DIR`
- `DATA_DIR`
- `MODELS_DIR`
- `MODEL_TIER=default|quality`
- `INDEXING_PROFILE=balanced`
- `RUNTIME_PROFILE`
- `GPU_BACKEND`
- `QDRANT_URL=local`
- `ALLOW_MOCK_MODELS=false`
- `AUTO_DOWNLOAD_MODELS=true`
- `HF_HOME=<models_dir>/huggingface`
- `ASR_LANGUAGE=auto`

Backend logs are written to `data/logs/backend.log`. Model selection, quantization, attention backend, device, latency, and memory information are emitted by the Python runtime logs.
