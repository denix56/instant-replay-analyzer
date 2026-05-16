# Model Preflight

Required preflight checks before running real inference:

- Exact model IDs exist on Hugging Face.
- All configured model families are below 10B parameters.
- `transformers` main, `torch`, `accelerate`, `qwen-vl-utils`, and bitsandbytes are importable in the selected environment.
- CUDA/ROCm use bitsandbytes NF4 4-bit for supported Qwen roles.
- CPU bitsandbytes is enabled only when the installed backend passes a smoke test.
- Mac/Metal uses compact unquantized Qwen3.5 2B behavior.
- Quality tier is rejected on CPU and Mac/Metal.
- Attention backend selection tries `flash_attention_3`, `flash_attention_2`, `sdpa`, then `eager`.
- On CUDA/ROCm installs using the optional accelerator extras, Qwen3.5 fast-path preflight checks `flash-linear-attention` (`fla`) and `causal-conv1d`; missing or failing packages leave Qwen3.5 on the PyTorch fallback path.
- Whisper ASR uses the Transformers ASR pipeline with 30-second chunking, timestamp chunks, and no forced language by default.
- MiDashengLM audio input is 16 kHz mono, chunked to at most 30 seconds, and sent through the model-card chat template with an audio content item.
- Qwen3-VL video preprocessing passes `video_metadata` and `do_resize=False` for Qwen3VL inputs.
- Qwen3-VL embeddings use the repository `scripts/qwen3_vl_embedding.py` `Qwen3VLEmbedder.process` path with direct source video paths for full-clip `video` embeddings; frame-sequence embeddings and timestamped window video embeddings are disabled.
- Qwen3-VL reranking uses the repository `scripts/qwen3_vl_reranker.py` `Qwen3VLReranker.process` path; unsupported APIs fail clearly.
