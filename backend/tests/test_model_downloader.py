import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.hf_pipeline.model_registry import ModelRegistryError
from backend.app.model_downloader import ModelSnapshotDownloadSpec, _valid_existing, ensure_models, model_specs_for_tier


def test_model_specs_for_default_tier_use_hf_snapshots(tmp_path):
    specs = model_specs_for_tier("default", tmp_path, gpu_backend="cuda")

    assert [spec.role for spec in specs] == [
        "asr",
        "summarizer",
        "audio_captioner",
        "embedder",
        "reranker",
    ]
    assert [spec.model_id for spec in specs] == [
        "openai/whisper-large-v3",
        "Qwen/Qwen3.5-2B",
        "mispeech/midashenglm-0.6b-fp32",
        "Qwen/Qwen3-VL-Embedding-2B",
        "Qwen/Qwen3-VL-Reranker-2B",
    ]
    assert all("*.safetensors" in spec.allow_patterns for spec in specs)


def test_model_specs_for_cpu_default_use_qwen35_2b(tmp_path):
    specs = model_specs_for_tier("default", tmp_path, gpu_backend="cpu")

    assert any(spec.model_id == "Qwen/Qwen3.5-2B" for spec in specs)
    assert any(spec.model_id == "openai/whisper-large-v3-turbo" for spec in specs)


def test_model_download_dry_run_reports_missing(tmp_path):
    results = ensure_models("default", tmp_path, gpu_backend="cuda", dry_run=True)

    assert results
    assert {result.status for result in results} == {"missing"}
    assert {result.role for result in results} >= {"asr", "audio_captioner", "summarizer", "embedder", "reranker"}


def test_model_download_dry_run_rejects_interrupted_snapshot(tmp_path):
    local_dir = tmp_path / "snapshots" / "Qwen--partial"
    local_dir.mkdir(parents=True)
    (local_dir / "config.json").write_text("{}", encoding="utf-8")
    (local_dir / "model.safetensors.index.json").write_text(
        '{"weight_map": {"layer": "model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    cache_dir = local_dir / ".cache" / "huggingface" / "download"
    cache_dir.mkdir(parents=True)
    (cache_dir / "model.safetensors.lock").write_text("", encoding="utf-8")
    (cache_dir / "model.safetensors.incomplete").write_text("", encoding="utf-8")

    spec = ModelSnapshotDownloadSpec(
        role="summarizer",
        model_id="Qwen/partial",
        local_dir=local_dir,
        allow_patterns=("*.json", "*.safetensors"),
    )

    assert _valid_existing(spec) is False


def test_model_specs_for_quality_tier_use_quality_models(tmp_path):
    specs = model_specs_for_tier("quality", tmp_path, gpu_backend="cuda")

    assert any(spec.model_id == "Qwen/Qwen3.5-9B" for spec in specs)
    assert any(spec.model_id == "openai/whisper-large-v3" for spec in specs)
    assert any(spec.model_id == "Qwen/Qwen3-VL-Embedding-8B" for spec in specs)
    assert any(spec.model_id == "Qwen/Qwen3-VL-Reranker-8B" for spec in specs)
    assert any(spec.model_id == "mispeech/midashenglm-0.6b-fp32" for spec in specs)


def test_quality_download_rejected_on_cpu(tmp_path):
    with pytest.raises(ModelRegistryError):
        model_specs_for_tier("quality", tmp_path, gpu_backend="cpu")
