# Architecture

## Native Runtime

The app is native-only across Windows, macOS, and Linux.

- Tauri + React runs the desktop UI.
- The Tauri backend layer performs host operations: folder picker, runtime probing, `uv` bootstrap, dependency sync, Python backend process start/stop, and log path reporting.
- FastAPI runs locally through `uv run python -m app serve`.
- SQLite stores metadata, settings, scan state, operation progress, transcripts, summaries, tags, and segment metadata.
- Qdrant runs in embedded/local Python-client mode and persists vectors under `data/qdrant`.
- Hugging Face model snapshots are downloaded automatically under `models`.

No Docker containers, Compose profiles, or sidecar services are required.

## Pipeline

Scanning walks the configured clips folder, filters supported video files, assigns first-level subfolder groups, compares path/size/mtime against SQLite scan state, and marks files as new, unchanged, changed, or missing.

Indexing reuses scan state. Unchanged clips are skipped unless forced. New or changed clips go through metadata extraction, adaptive audio-video windowing, Qwen3.5 video-aware evidence fusion, Whisper ASR transcript extraction, MiDashengLM uncertain audio captioning, Qwen3-VL embedding, local Qdrant writes, and SQLite state updates.

When multiple clips are indexed together, execution is batched by model role. The app prepares every active clip first, runs Whisper ASR across the full batch, runs MiDashengLM audio captioning across the full batch, runs Qwen3.5 video observation and fusion across the full batch, and only then runs Qwen3-VL embedding across the full batch. This keeps one model role active at a time and avoids clip-by-clip model load/unload churn.

Qwen3.5 video observation and Qwen3-VL full-clip video embeddings use the same prepared video payload. The app samples frames with PyAV in-process, caps the frame count using the shared video settings, tone-maps HDR/10-bit sources to SDR RGB, downscales cached PNG frames, and passes the resulting frame sequence to Qwen as a video input. Sampling is weighted instead of uniform: the first half uses lower temporal density, the second half gets roughly twice that density, and the 16-21s action window gets an additional bump for hit/kill evidence. This avoids unbounded raw-video decoding in `qwen-vl-utils` while keeping both model roles on the same visual evidence.

If death-screen OCR detects a death screen, the detected frame is passed as an image input to the final Qwen3.5 fusion prompt. The prompt limits that image to death-screen visual/OCR evidence and asks the model to cite it with the `video` source label and `death_screen_frame` window id.

Search embeds the query once, searches local Qdrant vector fields independently, normalizes scores per field, combines weighted late-fusion scores, reranks candidate matches, then returns every sorted result above the configured relevance threshold with timestamps, preview frames, snippets, tags, and matched modality.

The `video` vector field contains one full-clip Qwen3-VL video embedding per clip. It uses the shared prepared Qwen video frame sequence, not timestamped window vectors. Representative frames remain available only for preview/thumbnail fields and HUD/loadout/death-screen detection.

Deep analysis is explicit. Reasoning models are not run over every frame or segment during default indexing.

## Model Roles

- Qwen3.5 video fusion: model-generated visual evidence and structured summary from sampled windows.
- Qwen3-VL embeddings: separate vectors for full video, summary, speech, audio caption, metadata, and fused payloads.
- Whisper ASR: speech transcription with automatic language detection by default.
- MiDashengLM 0.6B: uncertain non-speech and mixed-audio caption evidence.
- Qwen3-VL Reranker: final search reranking through in-process Transformers.
- Qwen3.5: direct video-aware structured evidence fusion and selected reasoning path when enabled.
- EasyOCR: optional OCR fallback.

Runtime profiles change host acceleration only: CUDA, ROCm/HIP, macOS Metal/MPS, or CPU.
