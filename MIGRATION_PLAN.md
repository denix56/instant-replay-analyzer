# Migration Plan

Current migration target: in-process Hugging Face Transformers with bitsandbytes where supported.

Implemented direction:

- Replace external model process management with `TransformersModelManager`.
- Use exact Hugging Face model IDs in `backend/app/hf_pipeline/model_registry.py`.
- Use automatic Hugging Face snapshot downloads through `backend/app/model_downloader.py`.
- Preserve `default` and `quality` tiers, with compact Qwen3.5 2B behavior on Mac/CPU.
- Use `mispeech/midashenglm-0.6b-fp32` for non-speech and mixed-audio captions in every tier.
- Use `qwen-vl-utils` for direct video preparation for Qwen3-VL video embeddings and Qwen3.5 video-aware fusion.
- Preserve six-field late-fusion retrieval and Qwen3-VL reranking.
