from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from threading import Thread
from typing import Any, Callable, Sequence

import numpy as np

from ..hf_pipeline.model_registry import (
    HFModelSpec,
    dtype_for_backend,
    quantization_for_backend,
)
from ..llm.schemas import TranscriptionResult, TranscriptionSegment


LOGGER = logging.getLogger(__name__)

JSON_GENERATION_REPETITION_PENALTY = 1.08


class ModelRuntimeError(RuntimeError):
    """Raised when an in-process Transformers model cannot satisfy a request."""


@dataclass
class LoadedModel:
    spec: HFModelSpec
    model: Any
    processor: Any = None
    tokenizer: Any = None
    device: str = "cpu"
    dtype: str = "auto"
    quantization: str = "none"
    attention_backend: str = "eager"
    torch_compile_mode: str = "off"
    torch_compile_backend: str = "none"
    torch_compile_profile: str = "none"
    torch_compile_status: str = "off"
    torch_compile_target: str | None = None
    torch_compile_cache_status: str = "off"
    torch_compile_cache_path: str | None = None
    generation_cache_implementation: str = "auto"
    max_memory: dict[Any, str] | None = None


class TransformersModelManager:
    """Loads one Hugging Face model at a time to bound memory use."""

    def __init__(
        self,
        *,
        models_dir: str | Path,
        logs_dir: str | Path,
        gpu_backend: str = "unknown",
        one_model_at_a_time: bool = True,
        torch_compile_mode: str = "off",
        torch_compile_backend: str = "inductor",
        torch_compile_profile: str = "reduce-overhead",
        generation_cache_implementation: str = "auto",
    ) -> None:
        self.models_dir = Path(models_dir)
        self.logs_dir = Path(logs_dir)
        self.gpu_backend = gpu_backend
        self.one_model_at_a_time = one_model_at_a_time
        self.torch_compile_mode = _normalize_compile_mode(torch_compile_mode)
        self.torch_compile_backend = torch_compile_backend or "inductor"
        self.torch_compile_profile = torch_compile_profile or "reduce-overhead"
        self.generation_cache_implementation = _normalize_generation_cache_implementation(
            generation_cache_implementation
        )
        self._loaded: LoadedModel | None = None

    @property
    def loaded(self) -> LoadedModel | None:
        return self._loaded

    def unload(self) -> None:
        if self._loaded is not None:
            LOGGER.info("unloading model role=%s model_id=%s", self._loaded.spec.role, self._loaded.spec.model_id)
        self._loaded = None
        gc.collect()
        try:
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            LOGGER.debug("failed to clear accelerator cache", exc_info=True)

    def ensure_loaded(self, spec: HFModelSpec) -> LoadedModel:
        existing = self._loaded
        if existing and existing.spec.role == spec.role and existing.spec.model_id == spec.model_id:
            return existing
        if self.one_model_at_a_time:
            self.unload()
        start = time.perf_counter()
        loaded = self._load(spec)
        loaded.generation_cache_implementation = self.generation_cache_implementation
        self._maybe_compile(loaded)
        self._loaded = loaded
        LOGGER.info(
            "loaded transformers model role=%s model_id=%s loader=%s device=%s dtype=%s "
            "quantization=%s attention=%s generation_cache=%s torch_compile=%s compile_target=%s compile_cache=%s "
            "compile_cache_path=%s latency_sec=%.3f memory=%s",
            spec.role,
            spec.model_id,
            spec.loader,
            loaded.device,
            loaded.dtype,
            loaded.quantization,
            loaded.attention_backend,
            loaded.generation_cache_implementation,
            loaded.torch_compile_status,
            loaded.torch_compile_target,
            loaded.torch_compile_cache_status,
            loaded.torch_compile_cache_path,
            time.perf_counter() - start,
            _memory_summary(),
        )
        return loaded

    def _maybe_compile(self, loaded: LoadedModel) -> None:
        loaded.torch_compile_mode = self.torch_compile_mode
        loaded.torch_compile_backend = self.torch_compile_backend
        loaded.torch_compile_profile = self.torch_compile_profile
        if self.torch_compile_mode == "off":
            loaded.torch_compile_status = "off"
            return
        skip_reason = _compile_skip_reason(loaded.spec)
        if skip_reason is not None:
            loaded.torch_compile_status = skip_reason
            return
        torch = _torch()
        if not hasattr(torch, "compile"):
            message = "torch.compile is unavailable in the installed torch build."
            if self.torch_compile_mode == "on":
                raise ModelRuntimeError(message)
            loaded.torch_compile_status = "failed_unavailable"
            LOGGER.warning("torch.compile skipped for role=%s model_id=%s: %s", loaded.spec.role, loaded.spec.model_id, message)
            return
        try:
            owner, attr_name, target_name = _compile_target(loaded.model, loaded.spec)
            loaded.torch_compile_target = target_name
            _configure_torch_compile(torch)
            cache_path = self._compile_cache_path(loaded, torch)
            loaded.torch_compile_cache_path = str(cache_path)
            self._load_compile_cache_artifacts(torch, loaded, cache_path)
            compile_kwargs: dict[str, Any] = {
                "backend": self.torch_compile_backend,
                "fullgraph": False,
                "dynamic": False,
            }
            if self.torch_compile_profile and self.torch_compile_profile != "default":
                compile_kwargs["mode"] = self.torch_compile_profile
            module_compile = getattr(owner, "compile", None)
            if callable(module_compile):
                module_compile(**compile_kwargs)
            else:
                compiled = torch.compile(getattr(owner, attr_name), **compile_kwargs)
                setattr(owner, attr_name, compiled)
            loaded.torch_compile_status = "compiled"
            self._save_compile_cache_artifacts(torch, loaded, cache_path, force=True)
        except Exception as exc:  # noqa: BLE001 - compile support varies by backend/model/runtime.
            message = (
                f"torch.compile failed for role={loaded.spec.role} model_id={loaded.spec.model_id} "
                f"target={loaded.torch_compile_target or 'unknown'} backend={self.torch_compile_backend} "
                f"profile={self.torch_compile_profile}: {exc}"
            )
            if self.torch_compile_mode == "on":
                raise ModelRuntimeError(message) from exc
            loaded.torch_compile_status = "failed"
            LOGGER.warning(message)

    def _compile_cache_path(self, loaded: LoadedModel, torch: Any | None = None) -> Path:
        torch = torch or _torch()
        key = {
            "torch": str(getattr(torch, "__version__", "unknown")),
            "model_id": loaded.spec.model_id,
            "loader": loaded.spec.loader,
            "device": loaded.device,
            "dtype": loaded.dtype,
            "quantization": loaded.quantization,
            "attention_backend": loaded.attention_backend,
            "compile_backend": self.torch_compile_backend,
            "compile_profile": self.torch_compile_profile,
            "compile_target": loaded.torch_compile_target,
        }
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        model_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", loaded.spec.model_id).strip("-") or "model"
        target_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", loaded.torch_compile_target or "target").strip("-")
        return self.models_dir / "torch_compile_cache" / f"{model_name}-{target_name}-{digest}.bin"

    def _load_compile_cache_artifacts(self, torch: Any, loaded: LoadedModel, cache_path: Path) -> None:
        if not cache_path.is_file():
            loaded.torch_compile_cache_status = "miss"
            return
        compiler = getattr(torch, "compiler", None)
        load_cache_artifacts = getattr(compiler, "load_cache_artifacts", None) if compiler is not None else None
        if not callable(load_cache_artifacts):
            loaded.torch_compile_cache_status = "load_unavailable"
            return
        try:
            cache_info = load_cache_artifacts(cache_path.read_bytes())
            loaded.torch_compile_cache_status = "loaded"
            LOGGER.info(
                "loaded torch compile cache artifacts role=%s model_id=%s path=%s cache_info=%s",
                loaded.spec.role,
                loaded.spec.model_id,
                cache_path,
                cache_info,
            )
        except Exception:
            loaded.torch_compile_cache_status = "load_failed"
            LOGGER.warning(
                "failed to load torch compile cache artifacts role=%s model_id=%s path=%s",
                loaded.spec.role,
                loaded.spec.model_id,
                cache_path,
                exc_info=True,
            )

    def _save_compile_cache_artifacts(self, torch: Any, loaded: LoadedModel, cache_path: Path | None, *, force: bool = False) -> None:
        if loaded.torch_compile_status != "compiled" or cache_path is None:
            return
        if not force and loaded.torch_compile_cache_status == "saved":
            return
        compiler = getattr(torch, "compiler", None)
        save_cache_artifacts = getattr(compiler, "save_cache_artifacts", None) if compiler is not None else None
        if not callable(save_cache_artifacts):
            loaded.torch_compile_cache_status = "save_unavailable"
            return
        try:
            artifacts = save_cache_artifacts()
            if artifacts is None:
                if loaded.torch_compile_cache_status not in {"loaded", "saved"}:
                    loaded.torch_compile_cache_status = "no_artifacts"
                return
            serialized_artifacts = artifacts[0] if isinstance(artifacts, tuple) else artifacts
            if not serialized_artifacts:
                if loaded.torch_compile_cache_status not in {"loaded", "saved"}:
                    loaded.torch_compile_cache_status = "no_artifacts"
                return
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            tmp_path.write_bytes(serialized_artifacts)
            tmp_path.replace(cache_path)
            loaded.torch_compile_cache_status = "saved"
            LOGGER.info(
                "saved torch compile cache artifacts role=%s model_id=%s path=%s bytes=%d",
                loaded.spec.role,
                loaded.spec.model_id,
                cache_path,
                len(serialized_artifacts),
            )
        except Exception:
            loaded.torch_compile_cache_status = "save_failed"
            LOGGER.warning(
                "failed to save torch compile cache artifacts role=%s model_id=%s path=%s",
                loaded.spec.role,
                loaded.spec.model_id,
                cache_path,
                exc_info=True,
            )

    def _save_compile_cache_artifacts_after_execution(self, loaded: LoadedModel) -> None:
        if loaded.torch_compile_status != "compiled" or not loaded.torch_compile_cache_path:
            return
        self._save_compile_cache_artifacts(_torch(), loaded, Path(loaded.torch_compile_cache_path))

    def generate_chat(
        self,
        spec: HFModelSpec,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        chat_template_kwargs: dict[str, Any] | None = None,
        thinking_budget_tokens: int | None = None,
        stop_after_json: bool = False,
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        loaded = self.ensure_loaded(spec)
        inputs, input_token_count = _chat_inputs(loaded, messages, chat_template_kwargs or {})
        model = loaded.model
        torch = _torch()
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5) if temperature > 0 else None,
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
        cache_implementation = self.generation_cache_implementation
        if cache_implementation != "auto":
            generation_kwargs["cache_implementation"] = cache_implementation
        if stop_after_json:
            generation_kwargs["repetition_penalty"] = JSON_GENERATION_REPETITION_PENALTY
            generation_kwargs["renormalize_logits"] = True
            generation_kwargs["stopping_criteria"] = _json_stopping_criteria(loaded, input_token_count)
        with torch.inference_mode():
            if (
                thinking_budget_tokens is not None
                and thinking_budget_tokens > 0
                and (chat_template_kwargs or {}).get("enable_thinking") is True
            ):
                text = _generate_chat_with_thinking_budget(
                    loaded,
                    inputs,
                    input_token_count,
                    thinking_budget_tokens=int(thinking_budget_tokens),
                    answer_max_new_tokens=max_new_tokens,
                    generation_kwargs=generation_kwargs,
                    stream_callback=stream_callback,
                )
                self._save_compile_cache_artifacts_after_execution(loaded)
                if stop_after_json:
                    return _first_complete_json_object_text(text) or text
                return text
            output_ids = _generate_with_optional_streamer(
                loaded,
                inputs,
                generation_kwargs,
                stream_callback=stream_callback,
                phase="answer",
                redact_text=False,
            )
        self._save_compile_cache_artifacts_after_execution(loaded)
        generated = _generated_tokens_only(output_ids, inputs)
        decoder = loaded.processor or loaded.tokenizer
        texts = decoder.batch_decode(generated, skip_special_tokens=True)
        text = str(texts[0] if texts else "").strip()
        if stop_after_json:
            return _first_complete_json_object_text(text) or text
        return text

    def transcribe(self, spec: HFModelSpec, audio_path: str | Path, *, language: str = "auto") -> TranscriptionResult:
        loaded = self.ensure_loaded(spec)
        processor = loaded.processor
        if processor is None:
            raise ModelRuntimeError(f"{spec.model_id} did not load a processor for ASR")
        transformers = _transformers()
        torch = _torch()
        dtype = _resolve_dtype(loaded.dtype, torch)
        generate_kwargs: dict[str, Any] = _whisper_generate_kwargs(loaded.model)
        if language and language.strip().lower() not in {"", "auto", "none"}:
            generate_kwargs["language"] = language.strip().lower()
        max_input = spec.max_input or {}
        chunk_length_s = int(max_input.get("chunk_length_s", 30))
        return_timestamps = bool(max_input.get("return_timestamps", True))
        pipe = transformers.pipeline(
            "automatic-speech-recognition",
            model=loaded.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=max(1, int(spec.batch_size or 1)),
            dtype=dtype,
            device=_pipeline_device(loaded.device),
        )
        sample_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16000) or 16000)
        result = pipe(
            _audio_array_for_asr(audio_path, sample_rate=sample_rate),
            return_timestamps=return_timestamps,
            generate_kwargs=generate_kwargs,
        )
        if isinstance(result, list):
            result = result[0] if result else {}
        text = str(result.get("text", "") if isinstance(result, dict) else "").strip()
        language_result = result.get("language") if isinstance(result, dict) else None
        return TranscriptionResult(
            text=text,
            segments=_segments_from_chunks(result.get("chunks", []) if isinstance(result, dict) else [], text),
            language=str(language_result).strip() if language_result else generate_kwargs.get("language"),
            engine=spec.model_id,
            confidence=1.0 if text else 0.0,
        )

    def caption_audio(
        self,
        spec: HFModelSpec,
        audio_path: str | Path,
        *,
        prompt: str,
        max_new_tokens: int = 192,
    ) -> str:
        loaded = self.ensure_loaded(spec)
        processor = loaded.processor
        tokenizer = loaded.tokenizer or processor
        if processor is None:
            raise ModelRuntimeError(f"{spec.model_id} did not load a processor for audio captioning")
        if not hasattr(processor, "apply_chat_template"):
            raise ModelRuntimeError(
                f"{spec.model_id} processor does not expose apply_chat_template required by MiDashengLM."
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": str(Path(audio_path))},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=True,
            return_dict=True,
        )
        inputs = _move_inputs(inputs, loaded.device, dtype=getattr(loaded.model, "dtype", None))
        torch = _torch()
        sample_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 50,
            "temperature": 1.0,
            "repetition_penalty": 1.05,
        }
        with torch.inference_mode():
            output_ids = loaded.model.generate(**inputs, **sample_kwargs)
        self._save_compile_cache_artifacts_after_execution(loaded)
        generated = _generated_tokens_only(output_ids, inputs)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def embed(self, spec: HFModelSpec, values: Sequence[str | dict[str, Any]]) -> list[list[float]]:
        loaded = self.ensure_loaded(spec)
        vectors = [_embedding_vector(loaded, value) for value in values]
        self._save_compile_cache_artifacts_after_execution(loaded)
        return vectors

    def rerank(self, spec: HFModelSpec, query: str, documents: Sequence[str]) -> list[float]:
        loaded = self.ensure_loaded(spec)
        model = loaded.model
        prompt = "Retrieve clips or clip evidence relevant to the user's query."
        process = getattr(model, "process", None)
        if process is not None:
            payload = {
                "instruction": prompt,
                "query": {"text": query},
                "documents": [{"text": document} for document in documents],
                "fps": 1.0,
            }
            scores = _scores_to_list(process(payload), len(documents))
            self._save_compile_cache_artifacts_after_execution(loaded)
            return scores
        raise ModelRuntimeError(
            f"{spec.model_id} does not expose a supported Qwen3-VL reranker API "
            "(expected Qwen3VLReranker.process)."
        )

    def _load(self, spec: HFModelSpec) -> LoadedModel:
        transformers = _transformers()
        torch = _torch()
        device = _select_device(self.gpu_backend, torch)
        quantization = quantization_for_backend(spec, self.gpu_backend)
        effective_quantization = quantization
        dtype_name = dtype_for_backend(spec, self.gpu_backend)
        dtype = _resolve_dtype(dtype_name, torch)
        common = {
            "cache_dir": str(self.models_dir),
            "trust_remote_code": spec.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if quantization == "nf4_4bit":
            if _qwen_cpu_offload_dtype_enabled():
                common["dtype"] = dtype
                effective_quantization = "bf16_cpu_offload"
            elif _qwen_cpu_offload_8bit_enabled():
                common["quantization_config"] = _bnb_8bit_cpu_offload_config(torch)
                effective_quantization = "int8_fp32_cpu_offload"
            else:
                common["quantization_config"] = _bnb_config(torch)
            common["device_map"] = "auto"
            max_memory = _qwen_max_memory_from_env()
            if max_memory:
                common["max_memory"] = max_memory
        elif quantization == "int8_8bit":
            common["dtype"] = dtype
            common["quantization_config"] = _bnb_8bit_config(torch)
            common["device_map"] = "auto"
            max_memory = _qwen_max_memory_from_env()
            if max_memory:
                common["max_memory"] = max_memory
        elif device in {"cuda", "mps"}:
            common["dtype"] = dtype
            common["device_map"] = {"": device}
        else:
            common["dtype"] = dtype

        if spec.loader == "qwen_vl_embedding":
            return self._load_qwen_vl_embedder(spec, common, device, quantization, dtype_name)
        if spec.loader == "qwen_vl_reranker":
            return self._load_qwen_vl_reranker(spec, common, device, quantization, dtype_name)
        processor = _load_processor(transformers, spec, common)
        tokenizer = _load_tokenizer(transformers, spec, common)
        model_class = _model_class(transformers, spec)
        errors: list[str] = []
        for attention in _attention_candidates(spec, self.gpu_backend):
            kwargs = dict(common)
            if attention not in {"eager", ""}:
                kwargs["attn_implementation"] = attention
            try:
                model = model_class.from_pretrained(spec.model_id, **kwargs)
                if effective_quantization not in {"nf4_4bit", "int8_8bit", "int8_fp32_cpu_offload", "bf16_cpu_offload"} and device in {"cuda", "mps"} and hasattr(model, "to"):
                    model = model.to(device)
                if hasattr(model, "eval"):
                    model.eval()
                return LoadedModel(
                    spec=spec,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    device=device,
                    dtype=dtype_name,
                    quantization=effective_quantization,
                    attention_backend=attention,
                    max_memory=common.get("max_memory"),
                )
            except Exception as exc:  # noqa: BLE001 - try lower attention backends before failing.
                errors.append(f"{attention}: {exc}")
                if "flash" not in str(exc).lower() and "attn" not in str(exc).lower():
                    break
        raise ModelRuntimeError(
            f"Failed to load {spec.model_id} with {spec.loader}. "
            f"Tried attention backends: {'; '.join(errors)}"
        )

    def _load_qwen_vl_embedder(
        self,
        spec: HFModelSpec,
        common: dict[str, Any],
        device: str,
        quantization: str,
        dtype_name: str,
    ) -> LoadedModel:
        errors: list[str] = []
        model_source = _local_snapshot_or_model_id(self.models_dir, spec.model_id)
        script = _qwen_vl_embedding_script(model_source)
        max_input = spec.max_input or {}
        for attention in _attention_candidates(spec, self.gpu_backend):
            kwargs = _qwen_script_model_kwargs(common, attention)
            try:
                model = _Qwen3VLScriptEmbedder(
                    script,
                    model_source=model_source,
                    model_kwargs=kwargs,
                    device=device,
                    max_length=int(max_input.get("max_length", 8192)),
                    min_pixels=int(max_input.get("min_pixels", 4 * 32 * 32)),
                    max_pixels=int(max_input.get("max_pixels", max_input.get("video_max_pixels", 1280 * 720))),
                    total_pixels=int(max_input.get("total_pixels", 10 * 768 * 32 * 32)),
                    fps=float(max_input.get("video_fps", 1.0)),
                    num_frames=int(max_input.get("video_max_frames", 64)),
                    max_frames=int(max_input.get("video_max_frames", 64)),
                )
                return LoadedModel(
                    spec=spec,
                    model=model,
                    device=device,
                    dtype=dtype_name,
                    quantization=quantization,
                    attention_backend=attention,
                )
            except Exception as exc:  # noqa: BLE001 - try lower attention backends before failing.
                errors.append(f"{attention}: {exc}")
                if "flash" not in str(exc).lower() and "attn" not in str(exc).lower():
                    break
        raise ModelRuntimeError(f"Failed to load {spec.model_id} with Qwen3VLEmbedder. Tried attention backends: {'; '.join(errors)}")

    def _load_qwen_vl_reranker(
        self,
        spec: HFModelSpec,
        common: dict[str, Any],
        device: str,
        quantization: str,
        dtype_name: str,
    ) -> LoadedModel:
        errors: list[str] = []
        model_source = _local_snapshot_or_model_id(self.models_dir, spec.model_id)
        script = _qwen_vl_reranker_script(model_source)
        max_input = spec.max_input or {}
        for attention in _attention_candidates(spec, self.gpu_backend):
            kwargs = _qwen_script_model_kwargs(common, attention)
            try:
                model = _Qwen3VLRerankerScriptWrapper(
                    script,
                    model_source=model_source,
                    model_kwargs=kwargs,
                    device=device,
                    max_length=int(max_input.get("max_length", 8192)),
                    min_pixels=int(max_input.get("min_pixels", 4 * 32 * 32)),
                    max_pixels=int(max_input.get("max_pixels", max_input.get("video_max_pixels", 1280 * 720))),
                    total_pixels=int(max_input.get("total_pixels", 4 * 2 * 1280 * 32 * 32)),
                    fps=float(max_input.get("video_fps", 1.0)),
                    num_frames=int(max_input.get("video_max_frames", 64)),
                    max_frames=int(max_input.get("video_max_frames", 64)),
                )
                return LoadedModel(
                    spec=spec,
                    model=model,
                    device=device,
                    dtype=dtype_name,
                    quantization=quantization,
                    attention_backend=attention,
                )
            except Exception as exc:  # noqa: BLE001 - try lower attention backends before failing.
                errors.append(f"{attention}: {exc}")
                if "flash" not in str(exc).lower() and "attn" not in str(exc).lower():
                    break
        raise ModelRuntimeError(f"Failed to load {spec.model_id} with Qwen3VLReranker. Tried attention backends: {'; '.join(errors)}")


_GLOBAL_MANAGER: TransformersModelManager | None = None
_GLOBAL_KEY: tuple[str, str, str, bool, str, str, str, str] | None = None


def transformers_runtime_manager(
    *,
    models_dir: str | Path,
    logs_dir: str | Path,
    gpu_backend: str,
    one_model_at_a_time: bool = True,
    torch_compile_mode: str = "off",
    torch_compile_backend: str = "inductor",
    torch_compile_profile: str = "reduce-overhead",
    generation_cache_implementation: str = "auto",
) -> TransformersModelManager:
    global _GLOBAL_KEY, _GLOBAL_MANAGER
    normalized_compile_mode = _normalize_compile_mode(torch_compile_mode)
    normalized_cache_implementation = _normalize_generation_cache_implementation(generation_cache_implementation)
    key = (
        str(models_dir),
        str(logs_dir),
        gpu_backend,
        one_model_at_a_time,
        normalized_compile_mode,
        torch_compile_backend,
        torch_compile_profile,
        normalized_cache_implementation,
    )
    if _GLOBAL_MANAGER is None or _GLOBAL_KEY != key:
        _GLOBAL_MANAGER = TransformersModelManager(
            models_dir=models_dir,
            logs_dir=logs_dir,
            gpu_backend=gpu_backend,
            one_model_at_a_time=one_model_at_a_time,
            torch_compile_mode=normalized_compile_mode,
            torch_compile_backend=torch_compile_backend,
            torch_compile_profile=torch_compile_profile,
            generation_cache_implementation=normalized_cache_implementation,
        )
        _GLOBAL_KEY = key
    return _GLOBAL_MANAGER


def _normalize_generation_cache_implementation(value: str | None) -> str:
    normalized = (value or "auto").strip().lower().replace("-", "_")
    if normalized in {"", "auto", "none", "off", "false", "0"}:
        return "auto"
    if normalized in {"dynamic", "offloaded", "static", "quantized"}:
        return normalized
    return "auto"


def _normalize_compile_mode(value: str | None) -> str:
    normalized = (value or "off").strip().lower()
    if normalized in {"on", "true", "1", "yes"}:
        return "on"
    if normalized in {"auto", "try"}:
        return "auto"
    return "off"


def _transformers() -> Any:
    try:
        import transformers
    except ModuleNotFoundError as exc:
        raise ModelRuntimeError(
            "transformers is required for model execution. Run `uv sync` or install backend dependencies."
        ) from exc
    return transformers


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModelRuntimeError("torch is required for model execution.") from exc
    return torch


def _local_snapshot_or_model_id(models_dir: Path, model_id: str) -> str:
    local_dir = models_dir / "snapshots" / model_id.replace("/", "--")
    return str(local_dir) if local_dir.is_dir() else model_id


def _qwen_vl_embedding_script(model_source: str) -> Any:
    script_path = Path(model_source) / "scripts" / "qwen3_vl_embedding.py"
    if not script_path.is_file():
        raise ModelRuntimeError(
            "Qwen3-VL embedding requires the model repository script "
            f"`scripts/qwen3_vl_embedding.py`, but it was not found at {script_path}. "
            "Run `uv run python -m app.cli download-models --tier default` or re-download the selected tier."
        )
    module_name = f"_ira_qwen3_vl_embedding_{abs(hash(script_path))}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ModelRuntimeError(f"Failed to load Qwen3-VL embedding script from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _qwen_vl_reranker_script(model_source: str) -> Any:
    script_path = Path(model_source) / "scripts" / "qwen3_vl_reranker.py"
    if not script_path.is_file():
        raise ModelRuntimeError(
            "Qwen3-VL reranker requires the model repository script "
            f"`scripts/qwen3_vl_reranker.py`, but it was not found at {script_path}. "
            "Run `uv run python -m app.cli download-models --tier default` or re-download the selected tier."
        )
    module_name = f"_ira_qwen3_vl_reranker_{abs(hash(script_path))}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ModelRuntimeError(f"Failed to load Qwen3-VL reranker script from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _qwen_script_model_kwargs(common: dict[str, Any], attention: str) -> dict[str, Any]:
    kwargs = {
        key: value
        for key, value in common.items()
        if key not in {"cache_dir", "trust_remote_code", "device_map"}
    }
    if attention not in {"eager", ""}:
        kwargs["attn_implementation"] = attention
    return kwargs


class _Qwen3VLScriptEmbedder:
    """Qwen3-VL embedding wrapper aligned with the repository qwen3_vl_embedding.py script."""

    def __init__(
        self,
        script: Any,
        *,
        model_source: str,
        model_kwargs: dict[str, Any],
        device: str,
        max_length: int,
        min_pixels: int,
        max_pixels: int,
        total_pixels: int,
        fps: float,
        num_frames: int,
        max_frames: int,
    ) -> None:
        self.script = script
        self.device = device
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.model = script.Qwen3VLForEmbedding.from_pretrained(model_source, **model_kwargs)
        if "device_map" not in model_kwargs and device in {"cuda", "mps"} and hasattr(self.model, "to"):
            self.model = self.model.to(device)
        self.processor = script.Qwen3VLProcessor.from_pretrained(model_source, padding_side="right")
        if hasattr(self.model, "eval"):
            self.model.eval()

    def process(self, inputs: list[dict[str, Any]], normalize: bool = True) -> Any:
        conversations = [
            self._format_model_input(
                text=item.get("text"),
                image=item.get("image"),
                video=item.get("video"),
                instruction=item.get("instruction"),
                fps=item.get("fps"),
                max_frames=item.get("max_frames"),
            )
            for item in inputs
        ]
        processed = self._preprocess_inputs(conversations)
        target_device = _model_input_device(self.model, self.device)
        processed = {
            key: value.to(target_device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
        torch = _torch()
        with torch.inference_mode():
            outputs = self.model(**processed)
        embeddings = self._pooling_last(outputs.last_hidden_state, processed["attention_mask"])
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def _format_model_input(
        self,
        *,
        text: str | None = None,
        image: Any = None,
        video: str | list[Any] | None = None,
        instruction: str | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        if instruction:
            instruction = instruction.strip()
            if instruction and not _ends_with_punctuation(instruction):
                instruction = f"{instruction}."
        content: list[dict[str, Any]] = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction or "Represent the user's input."}]},
            {"role": "user", "content": content},
        ]
        if not text and not image and not video:
            content.append({"type": "text", "text": "NULL"})
            return conversation
        if video:
            if isinstance(video, list):
                content.append(
                    {
                        "type": "video",
                        "video": [_qwen_media_uri(str(item)) if isinstance(item, str) else item for item in video],
                        "sample_fps": float(fps or self.fps),
                        "raw_fps": float(fps or self.fps),
                        "max_frames": int(max_frames or self.max_frames),
                        "max_pixels": self.max_pixels,
                    }
                )
            elif isinstance(video, str):
                content.append(
                    {
                        "type": "video",
                        "video": _qwen_media_uri(video),
                        "fps": float(fps or self.fps),
                        "max_frames": int(max_frames or self.max_frames),
                    }
                )
            else:
                raise TypeError(f"Unrecognized video type: {type(video)}")
        if image:
            if not isinstance(image, str):
                image_content = image
            else:
                image_content = _qwen_media_uri(image)
            content.append(
                {
                    "type": "image",
                    "image": image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )
        if text:
            content.append({"type": "text", "text": text})
        return conversation

    def _preprocess_inputs(self, conversations: list[list[dict[str, Any]]]) -> dict[str, Any]:
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images, video_inputs, video_kwargs = self.script.process_vision_info(
            conversations,
            image_patch_size=_processor_image_patch_size(self.processor, 16),
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None
        return self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )

    @staticmethod
    def _pooling_last(hidden_state: Any, attention_mask: Any) -> Any:
        torch = _torch()
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]


class _Qwen3VLRerankerScriptWrapper:
    """Qwen3-VL reranker wrapper aligned with the repository qwen3_vl_reranker.py script."""

    def __init__(
        self,
        script: Any,
        *,
        model_source: str,
        model_kwargs: dict[str, Any],
        device: str,
        max_length: int,
        min_pixels: int,
        max_pixels: int,
        total_pixels: int,
        fps: float,
        num_frames: int,
        max_frames: int,
    ) -> None:
        self.inner = script.Qwen3VLReranker(
            model_name_or_path=model_source,
            max_length=max_length,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            fps=fps,
            num_frames=num_frames,
            max_frames=max_frames,
            **model_kwargs,
        )
        self.model = self.inner.model
        self.score_linear = self.inner.score_linear
        if device in {"cuda", "mps"} and hasattr(self.model, "to"):
            torch = _torch()
            target = torch.device(device)
            self.model = self.model.to(target)
            self.score_linear = self.score_linear.to(target).to(self.model.dtype)
            self.inner.model = self.model
            self.inner.score_linear = self.score_linear
            self.inner.device = target

    def process(self, payload: dict[str, Any]) -> list[float]:
        return [float(score) for score in self.inner.process(payload)]


def _model_input_device(model: Any, configured_device: str) -> str:
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return configured_device


def _processor_image_patch_size(processor: Any, fallback: int = 16) -> int:
    image_processor = getattr(processor, "image_processor", None)
    value = getattr(image_processor, "patch_size", None)
    try:
        return int(value or fallback)
    except (TypeError, ValueError):
        return int(fallback)


def _qwen_media_uri(value: str) -> str:
    if value.startswith(("http://", "https://", "file://", "oss://", "data:")):
        return value
    return "file://" + str(Path(value).expanduser().resolve())


def _ends_with_punctuation(value: str) -> bool:
    import unicodedata

    return bool(value) and unicodedata.category(value[-1]).startswith("P")


def _bnb_config(torch: Any) -> Any:
    try:
        from transformers import BitsAndBytesConfig
    except ModuleNotFoundError as exc:
        raise ModelRuntimeError("transformers BitsAndBytesConfig is required for NF4 quantization.") from exc
    try:
        import bitsandbytes as bnb  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModelRuntimeError(
            "bitsandbytes is required for NF4 quantization on this backend. "
            "Install the configured PyPI bitsandbytes package or disable quantization."
        ) from exc
    if not hasattr(torch, "bfloat16"):
        raise ModelRuntimeError("torch.bfloat16 is required for configured Qwen NF4 quantization.")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _bnb_8bit_cpu_offload_config(torch: Any) -> Any:
    try:
        from transformers import BitsAndBytesConfig
    except ModuleNotFoundError as exc:
        raise ModelRuntimeError("transformers BitsAndBytesConfig is required for bitsandbytes offload.") from exc
    try:
        import bitsandbytes as bnb  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModelRuntimeError(
            "bitsandbytes is required for int8 CPU offload. Install bitsandbytes or disable QWEN_CPU_OFFLOAD_8BIT."
        ) from exc
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )


def _bnb_8bit_config(torch: Any) -> Any:
    try:
        from transformers import BitsAndBytesConfig
    except ModuleNotFoundError as exc:
        raise ModelRuntimeError("transformers BitsAndBytesConfig is required for 8-bit quantization.") from exc
    try:
        import bitsandbytes as bnb  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModelRuntimeError(
            "bitsandbytes is required for 8-bit quantization on this backend. "
            "Install the configured PyPI bitsandbytes package or disable quantization."
        ) from exc
    return BitsAndBytesConfig(load_in_8bit=True)


def _qwen_cpu_offload_8bit_enabled() -> bool:
    return os.environ.get("QWEN_CPU_OFFLOAD_8BIT", "").strip().lower() in {"1", "true", "yes", "on"}


def _qwen_cpu_offload_dtype_enabled() -> bool:
    return os.environ.get("QWEN_CPU_OFFLOAD_DTYPE", "").strip().lower() in {"1", "true", "yes", "on"}


def _qwen_max_memory_from_env() -> dict[Any, str] | None:
    """Optional Accelerate max_memory map for experiments with partial CPU offload."""

    gpu_memory = os.environ.get("QWEN_GPU_MAX_MEMORY")
    cpu_memory = os.environ.get("QWEN_CPU_MAX_MEMORY")
    if not gpu_memory and not cpu_memory:
        return None
    max_memory: dict[Any, str] = {}
    if gpu_memory:
        max_memory[0] = gpu_memory.strip()
    if cpu_memory:
        max_memory["cpu"] = cpu_memory.strip()
    return max_memory or None


def _resolve_dtype(name: str, torch: Any) -> Any:
    normalized = str(name or "auto").lower()
    if normalized in {"bf16", "bfloat16"} and hasattr(torch, "bfloat16"):
        return torch.bfloat16
    if normalized in {"fp16", "float16"} and hasattr(torch, "float16"):
        return torch.float16
    if normalized in {"fp32", "float32"} and hasattr(torch, "float32"):
        return torch.float32
    return "auto"


def _select_device(gpu_backend: str, torch: Any) -> str:
    backend = gpu_backend.strip().lower()
    if backend in {"cuda", "rocm"}:
        if torch.cuda.is_available():
            return "cuda"
        raise ModelRuntimeError(f"GPU_BACKEND={gpu_backend} was requested but torch.cuda is not available.")
    if backend in {"macos-metal", "metal", "mps"}:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        raise ModelRuntimeError("Metal/MPS backend was requested but torch.backends.mps is not available.")
    return "cpu"


def _pipeline_device(device: str) -> str | int:
    if device == "cuda":
        return 0
    if device == "mps":
        return "mps"
    return "cpu"


def _audio_array_for_asr(audio_path: str | Path, *, sample_rate: int) -> dict[str, Any]:
    try:
        import librosa
    except ModuleNotFoundError as exc:
        raise ModelRuntimeError("librosa is required for Python-native ASR audio loading.") from exc
    array, loaded_sample_rate = librosa.load(str(audio_path), sr=int(sample_rate), mono=True)
    audio = np.asarray(array, dtype=np.float32)
    if audio.ndim != 1:
        audio = np.mean(audio, axis=0, dtype=np.float32)
    return {"array": audio, "sampling_rate": int(loaded_sample_rate)}


def _model_class(transformers: Any, spec: HFModelSpec) -> Any:
    if spec.loader == "transformers_speech_seq2seq":
        return transformers.AutoModelForSpeechSeq2Seq
    if spec.loader == "transformers_image_text_to_text":
        return transformers.AutoModelForImageTextToText
    return transformers.AutoModelForCausalLM if spec.loader == "transformers_causal_lm" else transformers.AutoModel


def _model_kwargs_for_library(common: dict[str, Any], attention: str) -> dict[str, Any]:
    kwargs = {
        key: value
        for key, value in common.items()
        if key not in {"cache_dir", "trust_remote_code"}
    }
    if attention not in {"eager", ""}:
        kwargs["attn_implementation"] = attention
    return kwargs


def _compile_skip_reason(spec: HFModelSpec) -> str | None:
    if spec.loader == "transformers_image_text_to_text":
        return "skipped_qwen35_video_dynamic_shapes"
    if spec.loader == "qwen_vl_embedding":
        return "skipped_qwen3_vl_embedding_dynamic_shapes"
    if spec.loader == "qwen_vl_reranker":
        return None
    return "skipped_unsupported_role"


def _compile_target(model: Any, spec: HFModelSpec) -> tuple[Any, str, str]:
    if spec.loader == "qwen_vl_reranker":
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "forward"):
            return inner, "forward", "model.forward"
    if hasattr(model, "forward"):
        return model, "forward", "forward"
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "forward"):
        return inner, "forward", "model.forward"
    raise ModelRuntimeError(
        f"{spec.model_id} does not expose a compile target for {spec.loader} "
        "(expected forward or model.forward)."
    )


def _configure_torch_compile(torch: Any) -> None:
    dynamo = getattr(torch, "_dynamo", None)
    config = getattr(dynamo, "config", None)
    if config is None:
        return
    if hasattr(config, "capture_dynamic_output_shape_ops"):
        config.capture_dynamic_output_shape_ops = True
    if hasattr(config, "cache_size_limit"):
        try:
            config.cache_size_limit = max(int(config.cache_size_limit), 1000)
        except Exception:
            config.cache_size_limit = 1000


def _load_processor(transformers: Any, spec: HFModelSpec, common: dict[str, Any]) -> Any:
    try:
        return transformers.AutoProcessor.from_pretrained(
            spec.model_id,
            cache_dir=common["cache_dir"],
            trust_remote_code=spec.trust_remote_code,
        )
    except Exception:
        LOGGER.debug("processor unavailable for %s", spec.model_id, exc_info=True)
        return None


def _load_tokenizer(transformers: Any, spec: HFModelSpec, common: dict[str, Any]) -> Any:
    try:
        return transformers.AutoTokenizer.from_pretrained(
            spec.model_id,
            cache_dir=common["cache_dir"],
            trust_remote_code=spec.trust_remote_code,
        )
    except Exception:
        LOGGER.debug("tokenizer unavailable for %s", spec.model_id, exc_info=True)
        return None


def _attention_candidates(spec: HFModelSpec, gpu_backend: str) -> list[str]:
    if gpu_backend not in {"cuda", "rocm"}:
        return ["sdpa", "eager"]
    available = []
    for backend in spec.attention_backends:
        if backend == "flash_attention_3" and not _flash_attention3_available():
            continue
        if backend == "flash_attention_2" and not _flash_attention2_available():
            continue
        available.append(backend)
    return available or ["eager"]


def _flash_attention3_available() -> bool:
    try:
        torch = _torch()
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            return False
        import flash_attn_interface  # noqa: F401

        return True
    except Exception:
        return False


def _flash_attention2_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def _chat_inputs(
    loaded: LoadedModel,
    messages: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
) -> tuple[Any, int]:
    processor = loaded.processor or loaded.tokenizer
    if processor is None:
        raise ModelRuntimeError(f"{loaded.spec.model_id} did not load a processor/tokenizer")
    prepared_messages = _normalize_message_media_paths(messages)
    text = _apply_chat_template(processor, prepared_messages, chat_template_kwargs)
    image_inputs = None
    video_inputs = None
    video_metadata = None
    video_kwargs: dict[str, Any] = {}
    if _contains_media(prepared_messages):
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs, video_kwargs = process_vision_info(
                prepared_messages,
                return_video_kwargs=True,
                return_video_metadata=True,
                image_patch_size=_processor_image_patch_size(
                    processor,
                    int((loaded.spec.max_input or {}).get("image_patch_size", 16)),
                ),
            )
            if video_inputs is not None and video_inputs and isinstance(video_inputs[0], tuple):
                videos, metadatas = zip(*video_inputs)
                video_inputs = list(videos)
                video_metadata = list(metadatas)
        except Exception as exc:
            raise ModelRuntimeError(f"qwen-vl-utils failed to prepare vision inputs: {exc}") from exc
    call_kwargs: dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
    if image_inputs is not None:
        call_kwargs["images"] = image_inputs
        call_kwargs["do_resize"] = False
    if video_inputs is not None:
        call_kwargs["videos"] = video_inputs
        call_kwargs["video_metadata"] = video_metadata
        call_kwargs["do_resize"] = False
        call_kwargs.update(video_kwargs)
    inputs = processor(**call_kwargs)
    inputs = _move_inputs(inputs, loaded.device)
    token_count = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
    return inputs, token_count


def _apply_chat_template(processor: Any, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate_chat_with_thinking_budget(
    loaded: LoadedModel,
    inputs: Any,
    input_token_count: int,
    *,
    thinking_budget_tokens: int,
    answer_max_new_tokens: int,
    generation_kwargs: dict[str, Any],
    stream_callback: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    torch = _torch()
    _emit_stream_event(
        stream_callback,
        {
            "event": "thinking_start",
            "phase": "thinking",
            "budget_tokens": int(thinking_budget_tokens),
            "redacted": True,
        },
    )
    first_ids = _generate_with_optional_streamer(
        loaded,
        inputs,
        {
            **generation_kwargs,
            "max_new_tokens": max(1, int(thinking_budget_tokens)),
        },
        stream_callback=stream_callback,
        phase="thinking",
        redact_text=True,
    )
    generated_token_ids = _generated_token_ids(first_ids, input_token_count)
    _emit_stream_event(
        stream_callback,
        {
            "event": "thinking_end",
            "phase": "thinking",
            "generated_tokens": len(generated_token_ids),
            "budget_tokens": int(thinking_budget_tokens),
            "redacted": True,
        },
    )
    im_end_id = _special_token_id(loaded, "<|im_end|>", 151645)
    think_end_id = _special_token_id(loaded, "</think>", 151668)
    if im_end_id in generated_token_ids:
        return _decode_after_thinking(loaded, generated_token_ids, think_end_id)

    continuation_ids = first_ids
    if think_end_id not in generated_token_ids:
        early_stopping_text = (
            "\n\n Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n"
            "</think>\n\n"
        )
        early_stopping_ids = _encode_text_ids(loaded, early_stopping_text)
        continuation_ids = torch.cat([first_ids, early_stopping_ids.to(first_ids.device)], dim=-1)
    continuation_inputs = _continuation_inputs(inputs, continuation_ids, torch)
    final_ids = _generate_with_optional_streamer(
        loaded,
        continuation_inputs,
        {
            **generation_kwargs,
            "max_new_tokens": max(1, int(answer_max_new_tokens)),
        },
        stream_callback=stream_callback,
        phase="answer",
        redact_text=False,
    )
    return _decode_after_thinking(loaded, _generated_token_ids(final_ids, input_token_count), think_end_id)


def _generate_with_optional_streamer(
    loaded: LoadedModel,
    inputs: Any,
    generation_kwargs: dict[str, Any],
    *,
    stream_callback: Callable[[dict[str, Any]], None] | None,
    phase: str,
    redact_text: bool,
) -> Any:
    if stream_callback is None:
        return loaded.model.generate(**inputs, **generation_kwargs)
    try:
        from transformers import TextIteratorStreamer
    except ModuleNotFoundError:
        _emit_stream_event(
            stream_callback,
            {
                "event": "stream_unavailable",
                "phase": phase,
                "reason": "transformers TextIteratorStreamer is unavailable",
            },
        )
        return loaded.model.generate(**inputs, **generation_kwargs)

    tokenizer = _text_tokenizer(loaded)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=0.25,
    )
    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def run_generate() -> None:
        try:
            result["output_ids"] = loaded.model.generate(**inputs, **generation_kwargs, streamer=streamer)
        except BaseException as exc:  # noqa: BLE001 - propagate generation failures after draining.
            error["exception"] = exc

    thread = Thread(target=run_generate, daemon=True)
    chunks = 0
    chars = 0
    _emit_stream_event(stream_callback, {"event": "stream_start", "phase": phase, "redacted": redact_text})
    thread.start()
    while thread.is_alive():
        try:
            chunk = next(streamer)
        except StopIteration:
            break
        except Empty:
            continue
        chunks, chars = _emit_stream_chunk(
            stream_callback,
            phase=phase,
            chunk=str(chunk),
            chunks=chunks,
            chars=chars,
            redact_text=redact_text,
        )
    while True:
        try:
            chunk = next(streamer)
        except (StopIteration, Empty):
            break
        chunks, chars = _emit_stream_chunk(
            stream_callback,
            phase=phase,
            chunk=str(chunk),
            chunks=chunks,
            chars=chars,
            redact_text=redact_text,
        )
    thread.join()
    _emit_stream_event(
        stream_callback,
        {
            "event": "stream_end",
            "phase": phase,
            "chunks": chunks,
            "chars": chars,
            "redacted": redact_text,
        },
    )
    if "exception" in error:
        raise error["exception"]
    return result.get("output_ids")


def _emit_stream_chunk(
    stream_callback: Callable[[dict[str, Any]], None] | None,
    *,
    phase: str,
    chunk: str,
    chunks: int,
    chars: int,
    redact_text: bool,
) -> tuple[int, int]:
    if not chunk:
        return chunks, chars
    next_chunks = chunks + 1
    next_chars = chars + len(chunk)
    event: dict[str, Any] = {
        "event": "stream_chunk",
        "phase": phase,
        "chunk_index": next_chunks,
        "chunk_chars": len(chunk),
        "chars": next_chars,
        "redacted": redact_text,
    }
    if redact_text:
        event["text"] = None
    else:
        event["text"] = chunk
    _emit_stream_event(stream_callback, event)
    return next_chunks, next_chars


def _emit_stream_event(
    stream_callback: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    if stream_callback is None:
        return
    try:
        stream_callback(event)
    except Exception:
        LOGGER.debug("stream callback failed for event=%s", event.get("event"), exc_info=True)


def _json_stopping_criteria(loaded: LoadedModel, input_token_count: int) -> Any:
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ModuleNotFoundError as exc:
        raise ModelRuntimeError("transformers StoppingCriteria is required for JSON-bounded generation.") from exc

    tokenizer = _text_tokenizer(loaded)

    class StopAfterJsonObject(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:  # noqa: ANN401
            try:
                generated = input_ids[0][input_token_count:]
                text = tokenizer.decode(generated, skip_special_tokens=True)
            except Exception:
                return False
            return _has_complete_json_object(str(text))

    return StoppingCriteriaList([StopAfterJsonObject()])


def _has_complete_json_object(text: str) -> bool:
    return _first_complete_json_object_text(text) is not None


def _first_complete_json_object_text(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for offset, char in enumerate(text[start:]):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : start + offset + 1].strip()
    return None


def _generated_token_ids(output_ids: Any, input_token_count: int) -> list[int]:
    if hasattr(output_ids, "tolist") and len(output_ids.shape) == 2:
        return [int(item) for item in output_ids[0][input_token_count:].tolist()]
    if hasattr(output_ids, "tolist"):
        values = output_ids.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(item) for item in values[input_token_count:]]
    return []


def _continuation_inputs(inputs: Any, input_ids: Any, torch: Any) -> dict[str, Any]:
    continued = dict(inputs)
    continued["input_ids"] = input_ids
    continued["attention_mask"] = torch.ones_like(input_ids, dtype=torch.int64)
    for key in ("mm_token_type_ids", "input_token_type_ids", "token_type_ids", "input_token_type"):
        value = continued.get(key)
        if hasattr(value, "shape") and len(value.shape) >= 2 and value.shape[-1] != input_ids.shape[-1]:
            pad_length = int(input_ids.shape[-1] - value.shape[-1])
            if pad_length > 0:
                padding_shape = (*value.shape[:-1], pad_length)
                padding = torch.zeros(padding_shape, dtype=value.dtype, device=input_ids.device)
                continued[key] = torch.cat([value.to(input_ids.device), padding], dim=-1)
    for key in ("position_ids", "cache_position"):
        continued.pop(key, None)
    return continued


def _encode_text_ids(loaded: LoadedModel, text: str) -> Any:
    tokenizer = _text_tokenizer(loaded)
    encoded = tokenizer([text], return_tensors="pt", return_attention_mask=False)
    return encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]


def _decode_after_thinking(loaded: LoadedModel, token_ids: list[int], think_end_id: int) -> str:
    try:
        index = len(token_ids) - token_ids[::-1].index(think_end_id)
    except ValueError:
        index = 0
    tokenizer = _text_tokenizer(loaded)
    return str(tokenizer.decode(token_ids[index:], skip_special_tokens=True)).strip("\n ")


def _special_token_id(loaded: LoadedModel, token: str, fallback: int) -> int:
    tokenizer = _text_tokenizer(loaded)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        value = convert(token)
        if isinstance(value, int) and value >= 0:
            return value
    return fallback


def _text_tokenizer(loaded: LoadedModel) -> Any:
    tokenizer = loaded.tokenizer or getattr(loaded.processor, "tokenizer", None) or loaded.processor
    if tokenizer is None:
        raise ModelRuntimeError(f"{loaded.spec.model_id} did not load a tokenizer for text generation.")
    return tokenizer


def _contains_media(messages: Sequence[dict[str, Any]]) -> bool:
    serialized = json.dumps(messages, ensure_ascii=True).lower()
    return any(marker in serialized for marker in ('"type": "image"', '"type": "video"', '"type": "image_url"'))


def _normalize_message_media_paths(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            copied["content"] = [_normalize_content_media_paths(item) for item in content]
        normalized.append(copied)
    return normalized


def _normalize_content_media_paths(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    copied = dict(item)
    for key in ("image", "video"):
        value = copied.get(key)
        if isinstance(value, str):
            copied[key] = _media_uri(value)
        elif isinstance(value, list):
            copied[key] = [_media_uri(entry) if isinstance(entry, str) else entry for entry in value]
    return copied


def _media_uri(value: str) -> str:
    if value.startswith(("http://", "https://", "file://", "data:", "oss://")):
        return value
    return "file://" + str(Path(value).expanduser().resolve())


def _move_inputs(inputs: Any, device: str, dtype: Any | None = None) -> Any:
    if device == "cpu":
        return inputs
    if hasattr(inputs, "to"):
        if dtype is not None:
            try:
                return inputs.to(device=device, dtype=dtype)
            except TypeError:
                pass
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in dict(inputs).items()
    }


def _embedding_vector(loaded: LoadedModel, value: str | dict[str, Any]) -> list[float]:
    model = loaded.model
    if hasattr(model, "process"):
        prepared = _embedding_payload(value, include_model_kwargs=True)
        return _normalize_vector(model.process([prepared])[0])
    raise ModelRuntimeError(
        f"{loaded.spec.model_id} does not expose a supported Qwen3-VL embedding API "
        "(expected Qwen3VLEmbedder.process)."
    )


def _embedding_payload(value: str | dict[str, Any], *, include_model_kwargs: bool = False) -> str | dict[str, Any]:
    if isinstance(value, str):
        return value
    payload: dict[str, Any] = {}
    text = str(value.get("text") or "").strip()
    instruction = str(value.get("instruction") or "").strip()
    if instruction and not include_model_kwargs:
        text = f"{instruction}\n{text}".strip()
    if text:
        payload["text"] = text
    if instruction and include_model_kwargs:
        payload["instruction"] = instruction
    if value.get("image_path"):
        payload["image"] = str(value["image_path"])
    if value.get("video_path"):
        payload["video"] = str(value["video_path"])
        if include_model_kwargs:
            payload["fps"] = float(value.get("video_fps") or value.get("fps") or 1.0)
            payload["max_frames"] = int(value.get("video_max_frames") or value.get("max_frames") or 64)
    if value.get("video_frames"):
        payload["video"] = [str(item) for item in value["video_frames"]]
        if include_model_kwargs:
            payload["fps"] = float(value.get("video_fps") or value.get("fps") or 1.0)
            payload["max_frames"] = int(value.get("video_max_frames") or value.get("max_frames") or len(payload["video"]))
    if value.get("frame_paths"):
        raise ModelRuntimeError("frame_paths are reserved for previews and HUD/death detection, not embeddings")
    if not payload:
        payload["text"] = ""
    return payload


def _scores_to_list(values: Any, count: int) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        values = [float(values)]
    output = [0.0] * count
    for index, value in enumerate(list(values or [])[:count]):
        if isinstance(value, dict):
            value = value.get("score", value.get("relevance_score", 0.0))
        if isinstance(value, (list, tuple)):
            value = value[-1] if value else 0.0
        try:
            output[index] = float(value)
        except (TypeError, ValueError):
            output[index] = 0.0
    return output


def _ranked_scores_to_list(values: Any, count: int) -> list[float]:
    output = [0.0] * count
    for fallback_index, row in enumerate(list(values or [])):
        if isinstance(row, dict):
            index = row.get("corpus_id", row.get("index", fallback_index))
            value = row.get("score", row.get("relevance_score", 0.0))
        else:
            index = getattr(row, "corpus_id", getattr(row, "index", fallback_index))
            value = getattr(row, "score", getattr(row, "relevance_score", 0.0))
        try:
            numeric_index = int(index)
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if 0 <= numeric_index < count:
            output[numeric_index] = numeric_value
    return output


def _normalize_vector(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    values = [float(item) for item in list(value)]
    norm = math.sqrt(sum(item * item for item in values))
    if norm == 0:
        return values
    return [item / norm for item in values]


def _generated_tokens_only(output_ids: Any, inputs: Any) -> Any:
    try:
        input_ids = inputs["input_ids"]
        input_token_count = int(input_ids.shape[-1])
        if not (hasattr(output_ids, "shape") and len(output_ids.shape) == 2):
            return output_ids
        output_token_count = int(output_ids.shape[-1])
        if output_token_count <= input_token_count:
            return output_ids
        if _output_starts_with_input(output_ids, input_ids, input_token_count):
            return output_ids[:, input_token_count:]
    except Exception:
        return output_ids
    return output_ids


def _output_starts_with_input(output_ids: Any, input_ids: Any, input_token_count: int) -> bool:
    prefix: Any = None
    comparison: Any = None
    try:
        if not (hasattr(input_ids, "shape") and len(input_ids.shape) == 2):
            return False
        prefix = output_ids[:, :input_token_count]
        comparison = input_ids
        if getattr(prefix, "device", None) != getattr(comparison, "device", None) and hasattr(comparison, "to"):
            comparison = comparison.to(prefix.device)
        torch = _torch()
        return bool(torch.equal(prefix, comparison))
    except Exception:
        try:
            if prefix is None or comparison is None:
                return False
            return prefix.tolist() == comparison.tolist()
        except Exception:
            return False


def _read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    audio_path = Path(path)
    with wave.open(str(audio_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ModelRuntimeError(f"Expected 16-bit PCM WAV audio, got sample width {sample_width}: {audio_path}")
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate


def _whisper_generate_kwargs(model: Any | None = None) -> dict[str, Any]:
    # Whisper reserves decoder start/task tokens inside max_target_positions.
    # In current Transformers, passing the model-card 448 directly through the
    # ASR pipeline can become 3 prompt tokens + 448 new tokens and fail.
    max_new_tokens = 445
    max_target_positions = getattr(getattr(model, "config", None), "max_target_positions", None)
    if max_target_positions is not None:
        try:
            max_new_tokens = max(1, int(max_target_positions) - 3)
        except (TypeError, ValueError):
            max_new_tokens = 445
    return {
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
        "condition_on_prev_tokens": False,
        "compression_ratio_threshold": 1.35,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }


def _segments_from_chunks(chunks: Any, fallback_text: str) -> list[TranscriptionSegment]:
    segments: list[TranscriptionSegment] = []
    for chunk in list(chunks or []):
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text", "")).strip()
        timestamp = chunk.get("timestamp") or chunk.get("timestamps") or (None, None)
        try:
            start = float(timestamp[0] if timestamp[0] is not None else 0.0)
            end = float(timestamp[1] if timestamp[1] is not None else start)
        except (TypeError, ValueError, IndexError):
            start = 0.0
            end = 0.0
        if text:
            segments.append(
                TranscriptionSegment(
                    start=round(max(0.0, start), 3),
                    end=round(max(start, end), 3),
                    text=text,
                    confidence=None,
                )
            )
    return segments or _segments_from_text(fallback_text)


def _segments_from_text(text: str) -> list[TranscriptionSegment]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences and text:
        sentences = [text]
    segments = []
    cursor = 0.0
    for sentence in sentences:
        duration = max(1.0, min(12.0, len(sentence.split()) * 0.45))
        segments.append(
            TranscriptionSegment(
                start=round(cursor, 3),
                end=round(cursor + duration, 3),
                text=sentence,
                confidence=1.0,
            )
        )
        cursor += duration
    return segments


def _memory_summary() -> str:
    try:
        torch = _torch()
        if torch.cuda.is_available():
            return f"cuda_allocated={torch.cuda.memory_allocated()} cuda_reserved={torch.cuda.memory_reserved()}"
    except Exception:
        pass
    return "unavailable"
