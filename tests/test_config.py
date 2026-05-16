import pytest

from backend.app.cli import build_parser
from backend.app.config import (
    DEFAULT_QWEN_REASONING_BUDGET_TOKENS,
    AppSettings,
    detect_gpu_backend,
    normalize_qwen_reasoning_mode,
)
from backend.app.operations.schemas import AppConfig


def test_app_config_defaults_are_local_first():
    config = AppConfig()
    assert config.host == "127.0.0.1"
    assert config.port == 8000


def test_create_app_wires_operation_manager():
    pytest.importorskip("fastapi")
    from backend.app.api.server import create_app

    app = create_app()
    assert app.title == "Instant Replay Analyzer API"
    assert hasattr(app.state, "operations")


def test_cli_help_paths_exist():
    parser = build_parser()
    help_text = parser.format_help()
    assert "serve" in help_text
    assert "index" in help_text
    assert "search" in help_text


def test_app_settings_asr_language_defaults_to_auto(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ASR_LANGUAGE", raising=False)

    assert AppSettings.from_env().asr_language == "auto"


def test_app_settings_model_tier_defaults_to_default_and_quality_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("MODEL_TIER", raising=False)

    assert AppSettings.from_env().model_tier == "default"

    monkeypatch.setenv("MODEL_TIER", "quality")
    assert AppSettings.from_env().model_tier == "quality"


def test_app_settings_reranking_defaults_enabled_and_ocr_env_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ENABLE_RERANKING", raising=False)
    monkeypatch.setenv("ENABLE_OCR", "false")

    settings = AppSettings.from_env()

    assert settings.enable_reranking is True
    assert not hasattr(settings, "enable_ocr")

    monkeypatch.setenv("ENABLE_RERANKING", "false")
    assert AppSettings.from_env().enable_reranking is False


def test_auto_gpu_detection_prefers_cuda_when_nvidia_tools_exist(monkeypatch):
    monkeypatch.setattr("backend.app.config.platform.system", lambda: "Linux")
    monkeypatch.setattr("backend.app.config.platform.machine", lambda: "x86_64")
    monkeypatch.setattr(
        "backend.app.config.shutil.which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )
    monkeypatch.setattr("backend.app.config.Path.exists", lambda self: False)

    assert detect_gpu_backend() == "cuda"


def test_auto_gpu_detection_prefers_metal_on_apple_silicon(monkeypatch):
    monkeypatch.setattr("backend.app.config.platform.system", lambda: "Darwin")
    monkeypatch.setattr("backend.app.config.platform.machine", lambda: "arm64")

    assert detect_gpu_backend() == "macos-metal"


def test_app_settings_auto_profile_detects_gpu_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RUNTIME_PROFILE", "auto")
    monkeypatch.delenv("GPU_BACKEND", raising=False)
    monkeypatch.setattr("backend.app.config.detect_gpu_backend", lambda: "rocm")

    assert AppSettings.from_env().gpu_backend == "rocm"


def test_app_settings_asr_language_env_override_is_parsed_and_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ASR_LANGUAGE", " EN ")

    settings = AppSettings.from_env()
    settings.persist()

    assert settings.asr_language == "en"
    assert "asr_language: en" in settings.config_path.read_text(encoding="utf-8")

    monkeypatch.delenv("ASR_LANGUAGE")
    assert AppSettings.from_env().asr_language == "en"


def test_app_settings_torch_compile_env_override_is_parsed_and_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TORCH_COMPILE_MODE", "on")
    monkeypatch.setenv("TORCH_COMPILE_BACKEND", "inductor")
    monkeypatch.setenv("TORCH_COMPILE_PROFILE", "default")

    settings = AppSettings.from_env()
    settings.persist()

    assert settings.torch_compile_mode == "on"
    assert settings.torch_compile_backend == "inductor"
    assert settings.torch_compile_profile == "default"
    persisted = settings.config_path.read_text(encoding="utf-8")
    assert "torch_compile_mode: 'on'" in persisted or "torch_compile_mode: on" in persisted
    assert "torch_compile_profile: default" in persisted


def test_qwen_reasoning_defaults_by_tier_and_allows_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("QWEN_REASONING_MODE", raising=False)
    monkeypatch.delenv("MODEL_TIER", raising=False)

    assert normalize_qwen_reasoning_mode(None) == "off"
    settings = AppSettings.from_env()
    assert settings.qwen_reasoning_mode == "off"
    assert settings.qwen_reasoning_budget_tokens == DEFAULT_QWEN_REASONING_BUDGET_TOKENS

    monkeypatch.setenv("MODEL_TIER", "quality")
    assert AppSettings.from_env().qwen_reasoning_mode == "low"

    monkeypatch.setenv("QWEN_REASONING_MODE", "off")
    assert AppSettings.from_env().qwen_reasoning_mode == "off"

    monkeypatch.setenv("QWEN_REASONING_MODE", "full")
    assert AppSettings.from_env().qwen_reasoning_mode == "full"

    monkeypatch.setenv("QWEN_REASONING_BUDGET_TOKENS", "8192")
    assert AppSettings.from_env().qwen_reasoning_budget_tokens == 8192
