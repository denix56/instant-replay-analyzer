# Hunt Clip Search

Local-first semantic search for short gameplay clips, built for 25-second NVIDIA Instant Replay clips from Hunt: Showdown.

The app is native-only. There is no Docker runtime. The Tauri desktop app runs on the host and starts the Python/FastAPI backend locally through `uv`. SQLite stores metadata, Qdrant embedded/local mode stores vectors under `./data/qdrant`, and Hugging Face model snapshots are downloaded into the local model cache.

## Requirements

- Windows, macOS, or Linux.
- Node.js 20+ for the Tauri frontend during development.
- Rust/Cargo for running or building the Tauri shell during development.
- Python 3.13.
- `uv`; the native app attempts to install `uv` on launch if it is missing.

FFmpeg is preferred when installed on the system. The Python package `imageio-ffmpeg` is also included so media extraction has a bundled fallback.

## Install Python Dependencies

Use `uv` from the repository root:

```bash
uv sync
uv run python -m app --help
```

The root [pyproject.toml](./pyproject.toml) is the install/build definition for the native backend package and its dependencies.

## Open The App

```bash
cd native-ui
npm install
npm run tauri:dev
```

On app launch, when `Auto-start native backend` is enabled, the Tauri layer:

1. Detects `uv`, Python, and FFmpeg.
2. Installs `uv` if missing.
3. Runs `uv sync --project <repo> --no-dev`.
4. On Linux/NVIDIA, installs `causal-conv1d` and `flash-linear-attention` with `uv pip install` after the base PyTorch/CUDA environment is synced.
5. Writes `.env` with the selected clips folder, runtime profile, model tier, local data directory, and local model directory.
6. Starts `uv run --project <repo> python -m app serve --host 127.0.0.1 --port 8000`.
7. Polls `http://127.0.0.1:8000/health`.
8. Downloads missing Hugging Face model snapshots in the backend startup/background path.

Backend logs are written to `./data/logs/backend.log`.

## Runtime Profiles

Profiles are native host profiles:

- `Auto`: chooses a host profile.
- `macOS / Apple Silicon`: uses the MPS backend with compact Qwen3.5 defaults.
- `NVIDIA`: uses CUDA with Transformers and bitsandbytes quantization where supported.
- `AMD`: uses ROCm/HIP with Transformers and bitsandbytes quantization where supported.
- `CPU`: portable fallback.

The logical pipeline stays the same across profiles. Model roles are loaded one at a time through an in-process Transformers manager to avoid OOM on constrained machines. CUDA/ROCm use NF4 4-bit quantization where supported; Mac uses compact unquantized defaults; CPU uses the compact default path and quantization only when preflight confirms support.

## First Run

1. Select a clips folder with the native folder picker.
2. Choose runtime profile and model tier.
3. Leave `Auto-start native backend` enabled.
4. Finish setup.
5. Use Indexing to scan/index clips.
6. Use Search for natural-language queries.

Default folder suggestions:

- Windows: `C:\Users\<username>\Videos\NVIDIA\Instant Replay`, `C:\Users\<username>\Videos\NVIDIA`, `C:\Users\<username>\Videos\Captures`, `C:\Users\<username>\Videos`
- Linux: `~/Videos/NVIDIA`, `~/Videos/Captures`, `~/Videos`, `./clips`
- macOS: `~/nvidia test videos` when present, `~/Movies`, `~/Movies/NVIDIA`, `~/Videos`, `./clips`

## Model Tiers

`default`:

- ASR: `openai/whisper-large-v3` on CUDA/ROCm, `openai/whisper-large-v3-turbo` on Mac/CPU
- Summarizer and video-aware fusion: `Qwen/Qwen3.5-2B` with 8-bit quantization on CUDA/ROCm/CPU, unquantized on Mac/Metal
- Non-speech and mixed-audio captioning: `mispeech/midashenglm-0.6b-fp32`
- Retrieval: `Qwen/Qwen3-VL-Embedding-2B`
- Reranking: `Qwen/Qwen3-VL-Reranker-2B`

`quality`:

- ASR: `openai/whisper-large-v3`
- Summarizer and video-aware fusion: `Qwen/Qwen3.5-9B`
- Non-speech and mixed-audio captioning: `mispeech/midashenglm-0.6b-fp32`
- Retrieval: `Qwen/Qwen3-VL-Embedding-8B`
- Reranking: `Qwen/Qwen3-VL-Reranker-8B`

Quality tier requires CUDA or ROCm. Mac and CPU use the default compact path.

Indexing behavior is configured separately with `INDEXING_PROFILE=fast|balanced|detailed`. The default profile is `balanced`: 2.0s windows, 1.0s stride, and 2 representative frames.

## Model Downloads

The backend automatically downloads configured Hugging Face model snapshots when `AUTO_DOWNLOAD_MODELS=true`.

Manual command:

```bash
uv run python -m app download-models --tier default
```

Models are stored under:

```text
models/
  snapshots/
  huggingface/
```

## Data Directory

```text
data/
  app.db
  config.yaml
  logs/
  qdrant/
  segments/
```

SQLite stores metadata, scan state, settings, operation progress, transcripts, tags, summaries, and segment metadata. Qdrant local mode stores vector collections for `av_segments`, `transcript_text`, and `clip_metadata`.

## CLI

```bash
uv run python -m app serve
uv run python -m app index --source "/path/to/clips"
uv run python -m app search "boss lair shotgun fight"
uv run python -m app analyze --clip-id 1
uv run python -m app reset-index
```

Every command supports `--help`.

## Search Examples

- `clips where I died after re-peeking`
- `headshot through smoke`
- `boss lair shotgun fight`
- `funny teammate chaos`
- `enemy visible in window before I noticed`
- `missed audio cue before death`
- `teammate said he is on the left`
- `dogs barking near extraction`

## Automatic Clip Descriptions

During indexing, every processed clip gets a short description. The summarizer uses:

- Qwen3.5 direct video evidence and fusion
- Whisper transcript text when speech is detected
- MiDashengLM non-speech or mixed-audio caption evidence marked uncertain
- filename, group/game, metadata, duration, and segment modality counts

Deep reasoning models are not run over every segment during normal indexing.

## Hunt Knowledge Pack

Release builds can include a Hunt wiki knowledge pack with normalized entities, source
attribution, reference images, and Qwen3-VL text embeddings through the same local embedding runtime:

```bash
python -m app build-hunt-knowledge-pack --output data/packs/hunt-knowledge-pack --refresh
```

See `docs/hunt_knowledge_pack.md` for pack contents and crawler constraints.

## Tests

```bash
uv run pytest -q
```

Real native media/model integration tests are opt-in:

```bash
RUN_FULL_PIPELINE=1 \
FULL_PIPELINE_CLIPS_DIR="/Users/denyssenkin/nvidia test videos" \
FULL_PIPELINE_PROFILE=macos \
FULL_PIPELINE_MODEL_TIER=default \
uv run pytest -q tests/test_full_pipeline_real.py -x
```

The strict tests set `ALLOW_MOCK_MODELS=false` and use local Qdrant storage.

## Troubleshooting

- Missing Rust/Cargo: install Rust with `rustup` before running `npm run tauri:dev`.
- Missing `uv`: the app tries to install it; manual install is `curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux or the official PowerShell installer on Windows.
- Missing Python: install Python 3.13.
- Slow CPU indexing: use `MODEL_TIER=default` and `INDEXING_PROFILE=fast`.
- Missing models: keep `AUTO_DOWNLOAD_MODELS=true` or run `uv run python -m app download-models --tier default`.
- Media extraction fails: install system FFmpeg or rely on the bundled `imageio-ffmpeg` fallback.

## License

Copyright 2026 Denys Senkin.

Licensed under the Apache License, Version 2.0. See [LICENSE](./LICENSE).
