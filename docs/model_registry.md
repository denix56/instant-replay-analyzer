# Model Registry

`backend/app/hf_pipeline/model_registry.py` is the authoritative model contract for local Hugging Face execution. It defines each role, tier, exact model ID, loader strategy, dtype, quantization policy, attention backend order, max input constraints, and platform behavior.

## Tiers

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

Quality tier is rejected on CPU and Mac/Metal so the app does not silently choose models that are likely to be unusable locally.

## Runtime

Models run in-process through `TransformersModelManager`. The manager loads one role at a time, unloads the previous role, clears accelerator caches, and logs model ID, loader, device, dtype, quantization mode, attention backend, latency, and memory.

CUDA and ROCm use configurable NF4 4-bit bitsandbytes quantization for Qwen roles where supported. The CUDA environment targets `torch==2.12.0` with CUDA 13.0 libraries and `bitsandbytes==0.49.2`. ROCm on Windows temporarily disables NF4 4-bit quantization until that backend passes preflight reliably. CPU can use bitsandbytes only after the backend preflight succeeds. Mac/Metal uses unquantized compact defaults.

Attention backend selection is ordered as `flash_attention_3`, `flash_attention_2`, `sdpa`, then `eager`. Unsupported attention kernels fall through to the next supported backend for the same model.

Qwen3.5's gated-delta fast path can additionally use `flash-linear-attention` and `causal-conv1d`. The CUDA extra installs them only on Linux x86_64 after the base PyTorch/CUDA 13.0 environment is present so the extension is selected against the active Python, PyTorch, CUDA, and CXX11 ABI. If they are absent or fail preflight, Qwen uses the standard torch implementation instead of crashing at model startup.

Qwen3-VL embedding models load through the model repository `scripts/qwen3_vl_embedding.py` `Qwen3VLEmbedder` path so video payload preparation matches Qwen's reference script. Qwen3.5 video fusion and Qwen3-VL full-clip `video` embeddings both receive the same prepared frame-sequence video payload from the app's PyAV sampler; representative-frame fallbacks and timestamped window video embeddings are disabled. Qwen3-VL reranker models load through the model repository `scripts/qwen3_vl_reranker.py` `Qwen3VLReranker` path. Both loaders receive the same quantization, dtype, device, and attention backend settings as the rest of the Hugging Face runtime. If a loaded model does not expose the expected `process` API, runtime execution fails clearly instead of falling back to generic hidden-state pooling or zero rerank scores. Production code does not use SentenceTransformers.

## Video Embeddings

Qwen3-VL video embeddings use the same prepared frame-sequence video payload as Qwen3.5 fusion. Qwen3VL video processor calls still pass `video_metadata` and `do_resize=False` through the reference Qwen3VL path, but the app bounds decode first by sampling and downscaling frames with PyAV. The timestamp sampler preserves the configured total frame cap while biasing density toward the second half of the clip and the 16-21s action window. The `video` embedding record stores the source video path as `payload_ref`; the returned vector represents the whole supplied full-clip prepared video payload. Representative frames are not embedding inputs and are reserved for previews plus HUD/loadout/death-screen detection.
