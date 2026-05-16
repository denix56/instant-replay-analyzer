from __future__ import annotations

import platform
from dataclasses import dataclass, replace
from typing import Literal


ModelRole = Literal["asr", "audio_captioner", "summarizer", "embedder", "reranker"]
ModelTier = Literal["default", "quality"]
DeviceBackend = Literal["cuda", "rocm", "macos-metal", "cpu", "unknown"]
LoaderKind = Literal[
    "transformers_speech_seq2seq",
    "transformers_image_text_to_text",
    "transformers_causal_lm",
    "qwen_vl_embedding",
    "qwen_vl_reranker",
]
QuantizationMode = Literal["nf4_4bit", "int8_8bit", "bf16", "fp16", "fp32", "none", "configurable"]
AttentionBackend = Literal["flash_attention_3", "flash_attention_2", "sdpa", "eager"]
VIDEO_MAX_PIXELS_QWEN_LIMIT = 256000
VIDEO_MAX_FRAMES_QWEN_DEFAULT = 80
VIDEO_FOCUS_MAX_PIXELS_QWEN_LIMIT = 320000
VIDEO_FOCUS_MAX_FRAMES_QWEN_DEFAULT = 80
VIDEO_OCR_MAX_PIXELS_QWEN_LIMIT = 600000
VIDEO_OCR_MAX_FRAMES_QWEN_DEFAULT = 50
VIDEO_EMBED_MAX_PIXELS_QWEN_LIMIT = 224000
VIDEO_EMBED_MAX_FRAMES_QWEN_DEFAULT = 64


class ModelRegistryError(RuntimeError):
    """Raised when configured Hugging Face model contracts cannot be satisfied."""


COMMON_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "*.py",
    "tokenizer.*",
    "merges.txt",
    "vocab.*",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
    "chat_template*.jinja",
)


@dataclass(frozen=True)
class HFModelSpec:
    role: ModelRole
    tier: ModelTier
    model_id: str
    loader: LoaderKind
    quantization: QuantizationMode = "none"
    dtype: str = "auto"
    device: str = "auto"
    batch_size: int = 1
    max_input: dict[str, float | int | str | bool | None] | None = None
    failure_behavior: str = "fail_clear"
    supports_multimodal: bool = False
    supports_4bit: bool = False
    supports_8bit: bool = False
    use_bfloat16: bool = False
    trust_remote_code: bool = False
    parameter_count_b: float | None = None
    attention_backends: tuple[AttentionBackend, ...] = (
        "flash_attention_3",
        "flash_attention_2",
        "sdpa",
        "eager",
    )
    allow_patterns: tuple[str, ...] = COMMON_ALLOW_PATTERNS

    @property
    def source_model_id(self) -> str:
        return self.model_id


@dataclass(frozen=True)
class ModelRoleConfig:
    role: ModelRole
    default: HFModelSpec
    quality: HFModelSpec
    compact: HFModelSpec | None = None

    @property
    def default_model_id(self) -> str:
        return self.default.model_id

    @property
    def quality_model_id(self) -> str:
        return self.quality.model_id

    def model(self, tier: ModelTier = "default") -> HFModelSpec:
        return self.quality if tier == "quality" else self.default


def _spec(
    role: ModelRole,
    tier: ModelTier,
    model_id: str,
    loader: LoaderKind,
    *,
    quantization: QuantizationMode = "none",
    dtype: str = "auto",
    batch_size: int = 1,
    max_input: dict[str, float | int | str | bool | None] | None = None,
    supports_multimodal: bool = False,
    supports_4bit: bool = False,
    supports_8bit: bool = False,
    use_bfloat16: bool = False,
    trust_remote_code: bool = False,
    parameter_count_b: float | None = None,
) -> HFModelSpec:
    return HFModelSpec(
        role=role,
        tier=tier,
        model_id=model_id,
        loader=loader,
        quantization=quantization,
        dtype=dtype,
        batch_size=batch_size,
        max_input=max_input,
        supports_multimodal=supports_multimodal,
        supports_4bit=supports_4bit,
        supports_8bit=supports_8bit,
        use_bfloat16=use_bfloat16,
        trust_remote_code=trust_remote_code,
        parameter_count_b=parameter_count_b,
    )


DEFAULT_MODEL_REGISTRY: dict[ModelRole, ModelRoleConfig] = {
    "asr": ModelRoleConfig(
        role="asr",
        default=_spec(
            "asr",
            "default",
            "openai/whisper-large-v3",
            "transformers_speech_seq2seq",
            dtype="fp16",
            batch_size=1,
            max_input={"force_language": False, "chunk_length_s": 30, "return_timestamps": True},
            parameter_count_b=1.55,
        ),
        quality=_spec(
            "asr",
            "quality",
            "openai/whisper-large-v3",
            "transformers_speech_seq2seq",
            dtype="fp16",
            batch_size=1,
            max_input={"force_language": False, "chunk_length_s": 30, "return_timestamps": True},
            parameter_count_b=1.55,
        ),
        compact=_spec(
            "asr",
            "default",
            "openai/whisper-large-v3-turbo",
            "transformers_speech_seq2seq",
            dtype="fp32",
            batch_size=1,
            max_input={"force_language": False, "chunk_length_s": 30, "return_timestamps": True},
            parameter_count_b=0.81,
        ),
    ),
    "summarizer": ModelRoleConfig(
        role="summarizer",
        default=_spec(
            "summarizer",
            "default",
            "Qwen/Qwen3.5-2B",
            "transformers_image_text_to_text",
            quantization="none",
            dtype="bf16",
            max_input={
                "video_fps": 6.0,
                "video_max_frames": VIDEO_MAX_FRAMES_QWEN_DEFAULT,
                "video_max_pixels": VIDEO_MAX_PIXELS_QWEN_LIMIT,
                "max_pixels": VIDEO_MAX_PIXELS_QWEN_LIMIT,
                "focus_video_max_frames": VIDEO_FOCUS_MAX_FRAMES_QWEN_DEFAULT,
                "focus_video_max_pixels": VIDEO_FOCUS_MAX_PIXELS_QWEN_LIMIT,
                "ocr_video_max_frames": VIDEO_OCR_MAX_FRAMES_QWEN_DEFAULT,
                "ocr_video_max_pixels": VIDEO_OCR_MAX_PIXELS_QWEN_LIMIT,
                "image_patch_size": 16,
            },
            supports_multimodal=True,
            use_bfloat16=True,
            parameter_count_b=2.0,
        ),
        quality=_spec(
            "summarizer",
            "quality",
            "Qwen/Qwen3.5-9B",
            "transformers_image_text_to_text",
            quantization="nf4_4bit",
            dtype="bf16",
            max_input={
                "video_fps": 6.0,
                "video_max_frames": VIDEO_MAX_FRAMES_QWEN_DEFAULT,
                "video_max_pixels": VIDEO_MAX_PIXELS_QWEN_LIMIT,
                "max_pixels": VIDEO_MAX_PIXELS_QWEN_LIMIT,
                "focus_video_max_frames": VIDEO_FOCUS_MAX_FRAMES_QWEN_DEFAULT,
                "focus_video_max_pixels": VIDEO_FOCUS_MAX_PIXELS_QWEN_LIMIT,
                "ocr_video_max_frames": VIDEO_OCR_MAX_FRAMES_QWEN_DEFAULT,
                "ocr_video_max_pixels": VIDEO_OCR_MAX_PIXELS_QWEN_LIMIT,
                "image_patch_size": 16,
            },
            supports_multimodal=True,
            supports_4bit=True,
            use_bfloat16=True,
            parameter_count_b=9.0,
        ),
        compact=_spec(
            "summarizer",
            "default",
            "Qwen/Qwen3.5-2B",
            "transformers_image_text_to_text",
            quantization="none",
            dtype="bf16",
            max_input={
                "video_fps": 6.0,
                "video_max_frames": VIDEO_MAX_FRAMES_QWEN_DEFAULT,
                "video_max_pixels": VIDEO_MAX_PIXELS_QWEN_LIMIT,
                "max_pixels": VIDEO_MAX_PIXELS_QWEN_LIMIT,
                "focus_video_max_frames": VIDEO_FOCUS_MAX_FRAMES_QWEN_DEFAULT,
                "focus_video_max_pixels": VIDEO_FOCUS_MAX_PIXELS_QWEN_LIMIT,
                "ocr_video_max_frames": VIDEO_OCR_MAX_FRAMES_QWEN_DEFAULT,
                "ocr_video_max_pixels": VIDEO_OCR_MAX_PIXELS_QWEN_LIMIT,
                "image_patch_size": 16,
            },
            supports_multimodal=True,
            use_bfloat16=True,
            parameter_count_b=2.0,
        ),
    ),
    "audio_captioner": ModelRoleConfig(
        role="audio_captioner",
        default=_spec(
            "audio_captioner",
            "default",
            "mispeech/midashenglm-0.6b-fp32",
            "transformers_causal_lm",
            dtype="fp32",
            batch_size=1,
            max_input={"sample_rate": 16000, "channels": 1, "max_chunk_sec": 30, "window_sec": 5, "stride_sec": 2.5},
            supports_multimodal=True,
            trust_remote_code=True,
            parameter_count_b=0.6,
        ),
        quality=_spec(
            "audio_captioner",
            "quality",
            "mispeech/midashenglm-0.6b-fp32",
            "transformers_causal_lm",
            dtype="fp32",
            batch_size=1,
            max_input={"sample_rate": 16000, "channels": 1, "max_chunk_sec": 30, "window_sec": 5, "stride_sec": 2.5},
            supports_multimodal=True,
            trust_remote_code=True,
            parameter_count_b=0.6,
        ),
    ),
    "embedder": ModelRoleConfig(
        role="embedder",
        default=_spec(
            "embedder",
            "default",
            "Qwen/Qwen3-VL-Embedding-2B",
            "qwen_vl_embedding",
            quantization="nf4_4bit",
            dtype="bf16",
            batch_size=2,
            max_input={
                "video_fps": 6.0,
                "video_max_frames": VIDEO_EMBED_MAX_FRAMES_QWEN_DEFAULT,
                "video_max_pixels": VIDEO_EMBED_MAX_PIXELS_QWEN_LIMIT,
                "max_pixels": VIDEO_EMBED_MAX_PIXELS_QWEN_LIMIT,
                "image_patch_size": 16,
            },
            supports_multimodal=True,
            supports_4bit=True,
            use_bfloat16=True,
            trust_remote_code=True,
            parameter_count_b=2.0,
        ),
        quality=_spec(
            "embedder",
            "quality",
            "Qwen/Qwen3-VL-Embedding-8B",
            "qwen_vl_embedding",
            quantization="nf4_4bit",
            dtype="bf16",
            batch_size=1,
            max_input={
                "video_fps": 6.0,
                "video_max_frames": VIDEO_EMBED_MAX_FRAMES_QWEN_DEFAULT,
                "video_max_pixels": VIDEO_EMBED_MAX_PIXELS_QWEN_LIMIT,
                "max_pixels": VIDEO_EMBED_MAX_PIXELS_QWEN_LIMIT,
                "image_patch_size": 16,
            },
            supports_multimodal=True,
            supports_4bit=True,
            use_bfloat16=True,
            trust_remote_code=True,
            parameter_count_b=8.0,
        ),
    ),
    "reranker": ModelRoleConfig(
        role="reranker",
        default=_spec(
            "reranker",
            "default",
            "Qwen/Qwen3-VL-Reranker-2B",
            "qwen_vl_reranker",
            quantization="nf4_4bit",
            dtype="bf16",
            batch_size=2,
            max_input={"max_length": 2048},
            supports_multimodal=True,
            supports_4bit=True,
            use_bfloat16=True,
            trust_remote_code=True,
            parameter_count_b=2.0,
        ),
        quality=_spec(
            "reranker",
            "quality",
            "Qwen/Qwen3-VL-Reranker-8B",
            "qwen_vl_reranker",
            quantization="nf4_4bit",
            dtype="bf16",
            batch_size=1,
            max_input={"max_length": 2048},
            supports_multimodal=True,
            supports_4bit=True,
            use_bfloat16=True,
            trust_remote_code=True,
            parameter_count_b=8.0,
        ),
    ),
}


def normalize_model_tier(value: str | None) -> ModelTier:
    normalized = (value or "default").strip().lower()
    if normalized in {"quality", "high", "heavy"}:
        return "quality"
    return "default"


def normalize_device_backend(value: str | None) -> DeviceBackend:
    normalized = (value or "unknown").strip().lower()
    if normalized in {"cuda", "nvidia"}:
        return "cuda"
    if normalized in {"rocm", "amd", "hip"}:
        return "rocm"
    if normalized in {"macos-metal", "metal", "mps", "macos"}:
        return "macos-metal"
    if normalized == "cpu":
        return "cpu"
    return "unknown"


def validate_tier_for_backend(tier: ModelTier, backend: str) -> None:
    normalized = normalize_device_backend(backend)
    if tier == "quality" and normalized in {"cpu", "macos-metal"}:
        raise ModelRegistryError(
            "MODEL_TIER=quality is not available on CPU or Metal. "
            "Use MODEL_TIER=default or run with CUDA/ROCm."
        )


def model_config(role: ModelRole) -> ModelRoleConfig:
    return DEFAULT_MODEL_REGISTRY[role]


def model_for_role(
    role: ModelRole,
    tier: ModelTier = "default",
    *,
    device_backend: str = "cuda",
) -> HFModelSpec:
    selected_tier = normalize_model_tier(tier)
    backend = normalize_device_backend(device_backend)
    validate_tier_for_backend(selected_tier, backend)
    config = DEFAULT_MODEL_REGISTRY[role]
    if role == "asr" and selected_tier == "default" and backend in {"cpu", "macos-metal"} and config.compact:
        return replace(config.compact, device=_device_for_backend(backend))
    if role == "summarizer" and selected_tier == "default" and backend in {"cpu", "macos-metal"} and config.compact:
        return replace(config.compact, device=_device_for_backend(backend))
    spec = config.model(selected_tier)
    if selected_tier == "quality" and backend in {"cpu", "macos-metal"}:
        validate_tier_for_backend(selected_tier, backend)
    return replace(spec, device=_device_for_backend(backend))


def tier_models(tier: ModelTier = "default", *, device_backend: str = "cuda") -> list[HFModelSpec]:
    selected_tier = normalize_model_tier(tier)
    return [
        model_for_role(role, selected_tier, device_backend=device_backend)
        for role in ("asr", "summarizer", "audio_captioner", "embedder", "reranker")
    ]


def registry_model_ids() -> list[str]:
    ids: list[str] = []
    for config in DEFAULT_MODEL_REGISTRY.values():
        ids.append(config.default.model_id)
        ids.append(config.quality.model_id)
        if config.compact is not None:
            ids.append(config.compact.model_id)
    return sorted(set(ids))


def loader_class_name(config: ModelRoleConfig | HFModelSpec) -> str:
    spec = config.default if isinstance(config, ModelRoleConfig) else config
    return {
        "transformers_speech_seq2seq": "AutoModelForSpeechSeq2Seq",
        "transformers_image_text_to_text": "AutoModelForImageTextToText",
        "transformers_causal_lm": "AutoModelForCausalLM",
        "qwen_vl_embedding": "Qwen3VLEmbedder",
        "qwen_vl_reranker": "Qwen3VLReranker",
    }[spec.loader]


def quantization_for_backend(spec: HFModelSpec, backend: str) -> QuantizationMode:
    normalized = normalize_device_backend(backend)
    if spec.quantization == "nf4_4bit":
        if not spec.supports_4bit:
            return "none"
        if normalized == "rocm" and platform.system().lower() == "windows":
            return "none"
        if normalized in {"cuda", "rocm", "cpu"}:
            return "nf4_4bit"
        return "none"
    if spec.quantization == "int8_8bit":
        if not spec.supports_8bit:
            return "none"
        if normalized in {"cuda", "rocm", "cpu"}:
            return "int8_8bit"
        return "none"
    return "none"


def dtype_for_backend(spec: HFModelSpec, backend: str) -> str:
    normalized = normalize_device_backend(backend)
    if quantization_for_backend(spec, normalized) in {"nf4_4bit", "int8_8bit"}:
        return "bf16" if spec.use_bfloat16 else spec.dtype
    if normalized == "cpu":
        return "bf16" if spec.use_bfloat16 else "fp32"
    if normalized == "macos-metal":
        if spec.role in {"asr", "audio_captioner"}:
            return "fp32"
        return "fp16"
    return spec.dtype


def _device_for_backend(backend: DeviceBackend) -> str:
    if backend in {"cuda", "rocm"}:
        return "cuda"
    if backend == "macos-metal":
        return "mps"
    if backend == "cpu":
        return "cpu"
    return "auto"
