from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

from .config import AppSettings
from .hf_pipeline.model_registry import ModelTier, model_for_role, normalize_model_tier
from .runtime.transformers_runtime import transformers_runtime_manager


def smoke_models(tier: str | ModelTier, settings: AppSettings) -> dict[str, Any]:
    selected = normalize_model_tier(str(tier))
    manager = transformers_runtime_manager(
        models_dir=settings.models_dir,
        logs_dir=settings.logs_dir,
        gpu_backend=settings.gpu_backend,
        one_model_at_a_time=True,
        torch_compile_mode=settings.torch_compile_mode,
        torch_compile_backend=settings.torch_compile_backend,
        torch_compile_profile=settings.torch_compile_profile,
        generation_cache_implementation=settings.qwen_cache_implementation,
    )
    audio_path = _smoke_wav(settings.data_dir / "smoke" / "silence.wav")
    results: list[dict[str, Any]] = []
    checks = [
        ("asr", lambda spec: manager.transcribe(spec, audio_path, language="auto").text),
        (
            "audio_captioner",
            lambda spec: manager.caption_audio(
                spec,
                audio_path,
                prompt="Describe only non-speech audio cues. Return one sentence.",
                max_new_tokens=32,
            ),
        ),
        (
            "summarizer",
            lambda spec: manager.generate_chat(
                spec,
                [{"role": "user", "content": [{"type": "text", "text": "Return JSON: {\"ok\": true}"}]}],
                max_new_tokens=32,
                chat_template_kwargs={"enable_thinking": False},
            ),
        ),
        ("embedder", lambda spec: len(manager.embed(spec, ["test"])[0])),
        ("reranker", lambda spec: manager.rerank(spec, "test", ["test document"])[0]),
    ]
    try:
        for role, check in checks:
            spec = model_for_role(role, selected, device_backend=settings.gpu_backend)  # type: ignore[arg-type]
            try:
                output = check(spec)
                loaded = manager.loaded
                results.append(
                    {
                        "role": role,
                        "status": "ok",
                        "model_id": spec.model_id,
                        "loader": spec.loader,
                        "device": loaded.device if loaded is not None else None,
                        "dtype": loaded.dtype if loaded is not None else None,
                        "quantization": loaded.quantization if loaded is not None else None,
                        "attention_backend": loaded.attention_backend if loaded is not None else None,
                        "generation_cache_implementation": (
                            loaded.generation_cache_implementation if loaded is not None else None
                        ),
                        "torch_compile_status": loaded.torch_compile_status if loaded is not None else None,
                        "torch_compile_target": loaded.torch_compile_target if loaded is not None else None,
                        "output_preview": str(output)[:120],
                    }
                )
            except Exception as exc:  # noqa: BLE001 - smoke report should include all attempted failures.
                results.append({"role": role, "status": "failed", "error": str(exc)})
                if not settings.allow_mock_models:
                    break
    finally:
        manager.unload()
    return {"tier": selected, "results": results}


def _smoke_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    sample_rate = 16_000
    frames = bytearray()
    for index in range(sample_rate):
        value = int(math.sin(index / sample_rate * math.pi * 2 * 220) * 1000)
        frames.extend(value.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))
    return path
