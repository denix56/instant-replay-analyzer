from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.model_downloader import ensure_models, model_specs_for_tier

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm"}
DEFAULT_CLIPS_DIR = Path("/Users/denyssenkin/nvidia test videos")


pytestmark = pytest.mark.full_pipeline


def _enabled() -> bool:
    return os.getenv("RUN_FULL_PIPELINE") == "1"


def _clips_dir() -> Path:
    return Path(os.getenv("FULL_PIPELINE_CLIPS_DIR", str(DEFAULT_CLIPS_DIR))).expanduser()


def _profile() -> str:
    return os.getenv("FULL_PIPELINE_PROFILE", "cpu")


def _gpu_backend_for_profile(profile: str) -> str:
    return {
        "cpu": "cpu",
        "macos": "macos-metal",
        "nvidia": "cuda",
        "amd": "rocm",
    }[profile]


def _videos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def _max_test_clips() -> int:
    return max(1, int(os.getenv("FULL_PIPELINE_MAX_CLIPS", "1")))


def _sampled_clips_dir(source: Path, tmp_path: Path) -> Path:
    selected = _videos(source)[: _max_test_clips()]
    sampled = tmp_path / "sampled-clips"
    sampled.mkdir(parents=True, exist_ok=True)
    for video in selected:
        target = sampled / video.name
        if target.exists():
            continue
        try:
            target.hardlink_to(video)
        except OSError:
            shutil.copy2(video, target)
    return sampled


def _model_files() -> dict[str, list[Path]]:
    models = REPO_ROOT / "models"
    backend = _gpu_backend_for_profile(_profile())
    return {
        f"{spec.role}:{spec.model_id}": list(spec.local_dir.rglob("config.json")) if spec.local_dir.is_dir() else []
        for spec in model_specs_for_tier(os.getenv("FULL_PIPELINE_MODEL_TIER", "default"), models, gpu_backend=backend)
    }


def _require_enabled() -> None:
    if not _enabled():
        pytest.skip("Set RUN_FULL_PIPELINE=1 to run real video/model integration tests.")


def test_real_full_pipeline_preflight() -> None:
    _require_enabled()
    clips_dir = _clips_dir()
    assert clips_dir.exists(), f"FULL_PIPELINE_CLIPS_DIR does not exist: {clips_dir}"
    assert clips_dir.is_dir(), f"FULL_PIPELINE_CLIPS_DIR is not a directory: {clips_dir}"

    videos = _videos(clips_dir)
    assert videos, (
        f"No supported videos found in {clips_dir}. Expected at least one "
        ".mp4, .mkv, .mov, or .webm file."
    )

    if os.getenv("AUTO_DOWNLOAD_MODELS", "1") != "0":
        results = ensure_models(
            os.getenv("FULL_PIPELINE_MODEL_TIER", "default"),
            REPO_ROOT / "models",
            gpu_backend=_gpu_backend_for_profile(_profile()),
            force=os.getenv("FORCE_MODEL_DOWNLOAD", "0") == "1",
        )
        failures = [result for result in results if result.status == "failed"]
        assert not failures, "\n".join(f"{item.role} {item.repo_id}: {item.error}" for item in failures)

    models = _model_files()
    missing = [name for name, files in models.items() if not files]
    assert not missing, (
        "Full-pipeline model tests require local Hugging Face snapshots under ./models. "
        f"Missing: {', '.join(missing)}. Expected examples: "
        "models/snapshots/<org--model>/config.json."
    )


def test_native_full_pipeline_indexes_and_searches_real_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_enabled()
    clips_dir = _clips_dir()
    videos = _videos(clips_dir)
    assert videos, f"No supported videos found in {clips_dir}"
    sampled_clips_dir = _sampled_clips_dir(clips_dir, tmp_path)

    profile = _profile()
    assert profile in {"cpu", "macos", "nvidia", "amd"}, f"Unsupported FULL_PIPELINE_PROFILE={profile}"

    data_dir = tmp_path / "data"
    for key, value in {
        "CLIPS_DIR": str(sampled_clips_dir),
        "DATA_DIR": str(data_dir),
        "MODELS_DIR": str(REPO_ROOT / "models"),
        "MODEL_TIER": os.getenv("FULL_PIPELINE_MODEL_TIER", "default"),
        "INDEXING_PROFILE": os.getenv("FULL_PIPELINE_INDEXING_PROFILE", "balanced"),
        "RUNTIME_PROFILE": profile,
        "GPU_BACKEND": _gpu_backend_for_profile(profile),
        "ALLOW_MOCK_MODELS": "false",
        "AUTO_DOWNLOAD_MODELS": "true",
        "QDRANT_URL": "local",
    }.items():
        monkeypatch.setenv(key, value)

    from backend.app.pipeline import run_indexing, run_search

    index_result = run_indexing(source=str(sampled_clips_dir), force=True)
    assert index_result["completed"] > 0, index_result
    assert index_result["failed"] == 0, index_result
    assert index_result["qdrant_active"] is True, index_result

    search_result = run_search(query=os.getenv("FULL_PIPELINE_QUERY", "boss lair shotgun fight"), limit=5)
    assert search_result["results"], search_result
    assert all("matched_modality" in result for result in search_result["results"])


def test_host_metal_full_pipeline_indexes_and_searches_real_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_enabled()
    if os.getenv("RUN_HOST_METAL_FULL_PIPELINE") != "1":
        pytest.skip("Set RUN_HOST_METAL_FULL_PIPELINE=1 to run host-side Apple Metal inference.")
    if sys.platform != "darwin":
        pytest.skip("Host Metal pipeline requires macOS.")

    clips_dir = _clips_dir()
    assert _videos(clips_dir), f"No supported videos found in {clips_dir}"
    sampled_clips_dir = _sampled_clips_dir(clips_dir, tmp_path)
    data_dir = tmp_path / "host-metal-data"
    models_dir = REPO_ROOT / "models"

    for key, value in {
        "DATA_DIR": str(data_dir),
        "MODELS_DIR": str(models_dir),
        "MODEL_TIER": os.getenv("FULL_PIPELINE_MODEL_TIER", "default"),
        "INDEXING_PROFILE": os.getenv("FULL_PIPELINE_INDEXING_PROFILE", "balanced"),
        "GPU_BACKEND": "macos-metal",
        "COMPOSE_PROFILE": "macos",
        "RUNTIME_PROFILE": "macos",
        "ALLOW_MOCK_MODELS": "false",
        "AUTO_DOWNLOAD_MODELS": "true",
        "QDRANT_URL": "local",
    }.items():
        monkeypatch.setenv(key, value)

    from backend.app.pipeline import run_indexing, run_search

    index_result = run_indexing(source=str(sampled_clips_dir), force=True)
    assert index_result["completed"] > 0, index_result
    assert index_result["failed"] == 0, index_result
    assert index_result["qdrant_active"] is True, index_result

    search_result = run_search(query=os.getenv("FULL_PIPELINE_QUERY", "player downed gunfight"), limit=3)
    assert search_result["results"], search_result
    assert search_result["fallback"] is False, search_result
    assert search_result["results"][0]["matched_modality"] in {
        "audio_video",
        "video_only",
        "audio_only",
        "transcript",
        "metadata",
        "hybrid",
    }
