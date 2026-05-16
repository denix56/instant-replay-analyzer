# Final Report

## Implemented

- Replaced external model-serving processes with in-process Hugging Face Transformers execution.
- Added `TransformersModelManager` with one-role-at-a-time loading, accelerator cache cleanup, selected-device logging, dtype logging, quantization logging, attention backend logging, latency logging, and memory logging.
- Replaced exact local executable/model artifact downloads with constrained Hugging Face snapshot downloads.
- Updated Tauri startup to set local Hugging Face cache paths and remove server-specific environment settings.
- Preserved six-field embeddings and late-fusion retrieval.
- Changed video embeddings to pass the shared prepared full-clip Qwen video payload; timestamped window video embeddings and representative-frame embedding inputs are disabled.
- Changed non-speech and mixed-audio evidence to `mispeech/midashenglm-0.6b-fp32` for every tier.
- Changed multi-clip indexing to staged model-role execution: prepare all clips, ASR all clips, audio-caption all clips, Qwen3.5 observation/fusion all clips, then Qwen3-VL embed all clips.
- Added detected death-screen frame image input to the final Qwen3.5 fusion prompt with a `death_screen_frame` evidence contract.
- Unified Qwen3.5 and Qwen3-VL embedding video preparation through an in-process PyAV sampler that bounds frame count, downscales frames, and tone-maps HDR/10-bit inputs to SDR RGB before either model role runs.

## Selected Models

- ASR default on CUDA/ROCm: `openai/whisper-large-v3`
- ASR compact on Mac/CPU: `openai/whisper-large-v3-turbo`
- ASR quality on CUDA/ROCm: `openai/whisper-large-v3`
- Summarizer default on CUDA/ROCm: `Qwen/Qwen3.5-2B` with 8-bit quantization
- Summarizer quality on CUDA/ROCm: `Qwen/Qwen3.5-9B`
- Summarizer compact on CPU/Mac: `Qwen/Qwen3.5-2B` with 8-bit quantization on CPU and unquantized on Mac/Metal
- Audio captions all tiers: `mispeech/midashenglm-0.6b-fp32`
- Embedding default: `Qwen/Qwen3-VL-Embedding-2B`
- Embedding quality: `Qwen/Qwen3-VL-Embedding-8B`
- Reranker default: `Qwen/Qwen3-VL-Reranker-2B`
- Reranker quality: `Qwen/Qwen3-VL-Reranker-8B`

## Verification

- `uv lock`: passed; pinned `transformers` main and bitsandbytes main.
- `python3 -m compileall -q backend/app`: passed.
- `PYTHONPATH=backend:. uv run pytest backend/tests tests -q`: passed, 182 passed, 3 skipped.
- `PYTHONPATH=backend:. uv run pytest backend/tests/test_pipeline_staged_indexing.py -q`: passed.
- Production reference scan for removed external-runtime strings across backend, tests, docs, README, Tauri source, and package manifests: passed.
- `cargo check`: not run because `cargo` is not installed in this environment.
