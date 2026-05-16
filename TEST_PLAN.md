# Test Plan

- Registry tests cover model IDs, loader classes, parameter bounds, platform overrides, quality rejection on CPU/Mac, and quantization policy.
- Runtime tests cover one-model-at-a-time loading, model reuse, Qwen3VL video metadata preprocessing, Whisper ASR pipeline chunking/timestamps, MiDashengLM audio chat-template input, Qwen3VLEmbedder frame-sequence sampling, and Qwen3VLReranker APIs.
- Adapter tests cover Whisper auto-language behavior, MiDashengLM uncertainty, 16 kHz mono audio validation including the 30-second boundary, Qwen3.5 video prompt shape, and empty-response repair.
- Fusion prompt tests cover detected death-screen frame image attachment and its `death_screen_frame` evidence contract.
- Video preparation tests cover PyAV-based frame sampling, downscaling, caching, and the shared prepared frame-sequence contract used by Qwen3.5 and Qwen3-VL embeddings.
- Embedding tests cover six fields, prepared video payload records, metadata `file_name`, deterministic fallback, no generic hidden-state pooling for Qwen3-VL embeddings, and one-vector video embedding behavior.
- Pipeline tests cover staged multi-clip indexing so each model-facing role processes the whole active batch before the next role starts.
- Retrieval tests cover per-field search, score normalization, late fusion, evidence preservation, fail-clear reranker API behavior, and final reranked result shape.
- Repo scan acceptance: production code and docs should not contain removed external-runtime references.
