# Known Limitations

- Real model loading depends on local model/runtime support. The app reports missing models instead of silently substituting unrelated families.
- CUDA and ROCm acceleration depend on the host PyTorch installation and driver stack.
- macOS acceleration uses PyTorch MPS/Metal where the selected model path supports it.
- Generic audio-video embeddings can retrieve sound-like moments, but reliable Hunt-specific event classification needs labeled gameplay data.
- OCR is optional and only runs on selected frames when enabled.
- Deep reasoning is explicit and does not run during default indexing.
