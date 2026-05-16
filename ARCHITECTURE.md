# Architecture

The app is native-only across Windows, macOS, and Linux. Tauri starts the Python/FastAPI backend through `uv`; SQLite stores metadata and operation state; embedded Qdrant stores vectors under `data/qdrant`.

Model execution is in-process through Hugging Face Transformers. `TransformersModelManager` loads one role at a time, unloads the previous model, clears accelerator caches, and logs model ID, loader, device, dtype, quantization, attention backend, latency, and memory.

Runtime profiles map to CUDA, ROCm/HIP, Mac Metal/MPS, or CPU. CUDA/ROCm use bitsandbytes NF4 4-bit quantization for supported Qwen roles. Mac uses compact unquantized defaults. CPU uses compact defaults and bitsandbytes only when preflight confirms support.

## Pipeline

Video, audio, and metadata are converted into clip manifests and adaptive windows. The pipeline creates Qwen3.5 video observations, Whisper ASR transcripts with automatic language detection, MiDashengLM uncertain non-speech captions, a strict fused summary, Qwen3-VL embeddings, late-fusion retrieval candidates, and Qwen3-VL reranked results.

Indexing multiple clips is staged by model role instead of running every model per clip. The batch first prepares windows and metadata for all clips, then runs ASR for all active clips, then non-speech audio captioning for all active clips, then Qwen3.5 video observation and fusion for all active clips, then Qwen3-VL embeddings for all active clips. Intermediate artifacts are materialized in the batch state and persisted to SQLite/vector storage where the existing schema stores them before the next stage starts. When death-screen OCR finds a frame, that exact frame is attached as direct image evidence to the final Qwen3.5 fusion prompt with a `death_screen_frame` evidence contract.

Six embedding fields are stored and searched independently: `video`, `summary`, `speech`, `audio_caption`, `metadata`, and `fused`. Qwen3.5 video fusion and Qwen3-VL full-clip video embeddings share the same bounded video preparation path: PyAV samples the source video in-process, tone-maps HDR/10-bit sources to SDR RGB, downscales cached frame images, and passes that frame sequence as a Qwen video payload. The sampler is intentionally non-uniform: it keeps a lower-density first half, doubles density after the midpoint, and adds extra density around 16-21s for late-clip hit/kill evidence while preserving the configured total frame budget. Timestamped window video embeddings are not written. Representative frames are reserved for previews and HUD/death-screen detection, not embedding or model evidence. Qwen3-VL reranking uses the model repository `scripts/qwen3_vl_reranker.py` `Qwen3VLReranker.process` path. Production code does not use SentenceTransformers.
