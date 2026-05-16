import pytest

from backend.app.hf_pipeline.model_registry import (
    DEFAULT_MODEL_REGISTRY,
    ModelRegistryError,
    dtype_for_backend,
    loader_class_name,
    model_for_role,
    quantization_for_backend,
    registry_model_ids,
    tier_models,
)
from backend.app.hf_pipeline import model_registry


def test_model_registry_selects_transformers_defaults() -> None:
    ids = set(registry_model_ids())

    assert "openai/whisper-large-v3-turbo" in ids
    assert "openai/whisper-large-v3" in ids
    assert "Qwen/Qwen3.5-2B" in ids
    assert "Qwen/Qwen3.5-9B" in ids
    assert "Qwen/Qwen3-VL-Embedding-2B" in ids
    assert "Qwen/Qwen3-VL-Reranker-2B" in ids
    assert "mispeech/midashenglm-0.6b-fp32" in ids


def test_qwen_video_input_defaults_use_reduced_qwen_pixel_budget() -> None:
    summarizer = model_for_role("summarizer", "default", device_backend="cuda")
    embedder = model_for_role("embedder", "default", device_backend="cuda")

    assert summarizer.max_input is not None
    assert summarizer.max_input["video_fps"] == 6.0
    assert summarizer.max_input["video_max_frames"] == 80
    assert summarizer.max_input["video_max_pixels"] == 256000
    assert summarizer.max_input["max_pixels"] == 256000
    assert summarizer.max_input["focus_video_max_frames"] == 80
    assert summarizer.max_input["focus_video_max_pixels"] == 320000
    assert summarizer.max_input["ocr_video_max_frames"] == 50
    assert summarizer.max_input["ocr_video_max_pixels"] == 600000

    assert embedder.max_input is not None
    assert embedder.max_input["video_fps"] == 6.0
    assert embedder.max_input["video_max_frames"] == 64
    assert embedder.max_input["video_max_pixels"] == 224000
    assert embedder.max_input["max_pixels"] == 224000


def test_registry_loader_classes_match_transformers_contract() -> None:
    assert loader_class_name(DEFAULT_MODEL_REGISTRY["summarizer"]) == "AutoModelForImageTextToText"
    assert loader_class_name(DEFAULT_MODEL_REGISTRY["asr"]) == "AutoModelForSpeechSeq2Seq"
    assert loader_class_name(DEFAULT_MODEL_REGISTRY["audio_captioner"]) == "AutoModelForCausalLM"
    assert loader_class_name(DEFAULT_MODEL_REGISTRY["embedder"]) == "Qwen3VLEmbedder"
    assert loader_class_name(DEFAULT_MODEL_REGISTRY["reranker"]) == "Qwen3VLReranker"


def test_cpu_and_metal_use_qwen35_compact_summarizer() -> None:
    assert model_for_role("summarizer", "default", device_backend="cuda").model_id == "Qwen/Qwen3.5-2B"
    assert model_for_role("summarizer", "default", device_backend="cpu").model_id == "Qwen/Qwen3.5-2B"
    assert model_for_role("summarizer", "default", device_backend="macos-metal").model_id == "Qwen/Qwen3.5-2B"


def test_asr_uses_large_v3_on_gpu_and_turbo_only_on_cpu_or_metal() -> None:
    assert model_for_role("asr", "default", device_backend="cuda").model_id == "openai/whisper-large-v3"
    assert model_for_role("asr", "default", device_backend="rocm").model_id == "openai/whisper-large-v3"
    assert model_for_role("asr", "quality", device_backend="cuda").model_id == "openai/whisper-large-v3"
    assert model_for_role("asr", "default", device_backend="cpu").model_id == "openai/whisper-large-v3-turbo"
    assert model_for_role("asr", "default", device_backend="macos-metal").model_id == "openai/whisper-large-v3-turbo"


def test_quality_tier_rejected_on_cpu_and_metal() -> None:
    with pytest.raises(ModelRegistryError, match="MODEL_TIER=quality"):
        model_for_role("summarizer", "quality", device_backend="cpu")
    with pytest.raises(ModelRegistryError, match="MODEL_TIER=quality"):
        tier_models("quality", device_backend="macos-metal")


def test_audio_captioner_uses_midashenglm_06b_for_all_tiers() -> None:
    default = model_for_role("audio_captioner", "default")
    quality = model_for_role("audio_captioner", "quality")

    assert default.model_id == "mispeech/midashenglm-0.6b-fp32"
    assert quality.model_id == "mispeech/midashenglm-0.6b-fp32"
    assert default.dtype == "fp32"
    assert quality.dtype == "fp32"
    assert dtype_for_backend(default, "cuda") == "fp32"
    assert dtype_for_backend(default, "rocm") == "fp32"
    assert dtype_for_backend(default, "cpu") == "fp32"
    assert default.supports_4bit is False
    assert quantization_for_backend(default, "cuda") == "none"
    assert quantization_for_backend(default, "rocm") == "none"
    assert quantization_for_backend(default, "cpu") == "none"
    assert default.max_input is not None
    assert default.max_input["sample_rate"] == 16000
    assert default.max_input["channels"] == 1
    assert default.max_input["max_chunk_sec"] == 30
    assert default.max_input["window_sec"] == 5
    assert default.max_input["stride_sec"] == 2.5


def test_qwen_quantization_contract_by_backend() -> None:
    spec = model_for_role("embedder", "default", device_backend="cuda")

    assert quantization_for_backend(spec, "cuda") == "nf4_4bit"
    assert quantization_for_backend(spec, "rocm") == "nf4_4bit"
    assert quantization_for_backend(spec, "cpu") == "nf4_4bit"
    assert quantization_for_backend(spec, "macos-metal") == "none"
    assert dtype_for_backend(spec, "cuda") == "bf16"


def test_qwen35_default_summarizer_uses_unquantized_bf16() -> None:
    spec = model_for_role("summarizer", "default", device_backend="cuda")

    assert spec.model_id == "Qwen/Qwen3.5-2B"
    assert spec.quantization == "none"
    assert spec.supports_8bit is False
    assert spec.supports_4bit is False
    assert quantization_for_backend(spec, "cuda") == "none"
    assert quantization_for_backend(spec, "rocm") == "none"
    assert quantization_for_backend(spec, "cpu") == "none"
    assert quantization_for_backend(spec, "macos-metal") == "none"
    assert dtype_for_backend(spec, "cuda") == "bf16"


def test_rocm_on_windows_disables_4bit_quantization(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = model_for_role("embedder", "default", device_backend="rocm")

    monkeypatch.setattr(model_registry.platform, "system", lambda: "Windows")
    assert quantization_for_backend(spec, "rocm") == "none"
    assert dtype_for_backend(spec, "rocm") == "bf16"

    monkeypatch.setattr(model_registry.platform, "system", lambda: "Linux")
    assert quantization_for_backend(spec, "rocm") == "nf4_4bit"


def test_asr_and_audio_captioner_use_fp32_on_metal_for_stability() -> None:
    spec = model_for_role("asr", "default", device_backend="macos-metal")
    captioner = model_for_role("audio_captioner", "default", device_backend="macos-metal")

    assert dtype_for_backend(spec, "macos-metal") == "fp32"
    assert dtype_for_backend(captioner, "macos-metal") == "fp32"


def test_asr_language_is_not_forced_by_default() -> None:
    spec = model_for_role("asr")

    assert spec.max_input is not None
    assert spec.max_input["force_language"] is False
    assert spec.max_input["return_timestamps"] is True
    assert quantization_for_backend(spec, "cuda") == "none"
    assert quantization_for_backend(spec, "cpu") == "none"


def test_configured_source_models_remain_under_10b() -> None:
    for config in DEFAULT_MODEL_REGISTRY.values():
        for spec in [config.default, config.quality, config.compact]:
            if spec is None:
                continue
            assert spec.parameter_count_b is None or spec.parameter_count_b < 10
