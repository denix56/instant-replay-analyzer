# Repository Map

Key model runtime files:

- `backend/app/hf_pipeline/model_registry.py`: authoritative Hugging Face model registry.
- `backend/app/runtime/transformers_runtime.py`: in-process model manager and low-level inference helpers.
- `backend/app/hf_pipeline/adapters.py`: video observation, audio captioning, and fusion summarization adapters.
- `backend/app/embeddings/hf_multimodal_embedder.py`: Qwen3-VL embedding facade and backend.
- `backend/app/processing/transcription.py`: Whisper ASR facade and backend.
- `backend/app/search/reranking.py`: Qwen3-VL reranking facade.
- `backend/app/model_downloader.py`: Hugging Face snapshot download support.
- `backend/app/pipeline.py`: indexing/search wiring.
- `native-ui/src-tauri/src/lib.rs`: native backend bootstrap and environment writing.
