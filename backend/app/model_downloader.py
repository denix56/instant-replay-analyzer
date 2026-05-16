from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .hf_pipeline.model_registry import (
    HFModelSpec,
    ModelTier,
    normalize_model_tier,
    tier_models,
)


DownloadRole = Literal["asr", "audio_captioner", "summarizer", "embedder", "reranker"]


@dataclass(frozen=True)
class ModelSnapshotDownloadSpec:
    role: DownloadRole
    model_id: str
    local_dir: Path
    allow_patterns: tuple[str, ...]
    revision: str = "main"


@dataclass(frozen=True)
class ModelDownloadResult:
    role: str
    repo_id: str
    local_dir: str
    status: str
    files: list[str]
    error: str | None = None


def model_specs_for_tier(
    tier: str | ModelTier,
    models_dir: str | Path,
    *,
    gpu_backend: str = "cuda",
) -> list[ModelSnapshotDownloadSpec]:
    normalized = normalize_model_tier(str(tier))
    specs: list[ModelSnapshotDownloadSpec] = []
    for spec in tier_models(normalized, device_backend=gpu_backend):
        specs.append(_download_spec(spec, models_dir))
    return specs


def ensure_models(
    tier: str | ModelTier,
    models_dir: str | Path,
    *,
    gpu_backend: str = "cuda",
    force: bool = False,
    dry_run: bool = False,
) -> list[ModelDownloadResult]:
    results: list[ModelDownloadResult] = []
    for spec in model_specs_for_tier(tier, models_dir, gpu_backend=gpu_backend):
        try:
            if _valid_existing(spec) and not force:
                results.append(_result(spec, "present", _snapshot_files(spec.local_dir)))
                continue
            if dry_run:
                results.append(_result(spec, "missing", []))
                continue
            local_dir = _download_snapshot(spec, force=force)
            results.append(_result(spec, "downloaded", _snapshot_files(local_dir)))
        except Exception as exc:  # noqa: BLE001 - return structured failures to API/CLI.
            results.append(_result(spec, "failed", [], error=str(exc)))
    return results


def has_required_models(
    tier: str | ModelTier,
    models_dir: str | Path,
    *,
    gpu_backend: str = "cuda",
) -> bool:
    return all(_valid_existing(spec) for spec in model_specs_for_tier(tier, models_dir, gpu_backend=gpu_backend))


def _download_spec(spec: HFModelSpec, models_dir: str | Path) -> ModelSnapshotDownloadSpec:
    safe_name = spec.model_id.replace("/", "--")
    return ModelSnapshotDownloadSpec(
        role=spec.role,
        model_id=spec.model_id,
        local_dir=Path(models_dir) / "snapshots" / safe_name,
        allow_patterns=spec.allow_patterns,
    )


def _download_snapshot(spec: ModelSnapshotDownloadSpec, *, force: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic model downloads. "
            "Install backend requirements or run with AUTO_DOWNLOAD_MODELS=false."
        ) from exc

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    spec.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec.model_id,
        revision=spec.revision,
        local_dir=spec.local_dir,
        cache_dir=str(spec.local_dir.parent / ".cache"),
        allow_patterns=list(spec.allow_patterns),
        force_download=force,
        token=token,
    )
    return spec.local_dir


def _valid_existing(spec: ModelSnapshotDownloadSpec) -> bool:
    if not spec.local_dir.is_dir():
        return False
    if not any(path.name == "config.json" for path in spec.local_dir.rglob("config.json")):
        return False
    if any(path.suffix in {".incomplete", ".lock"} for path in (spec.local_dir / ".cache").rglob("*")):
        return False
    if not _has_complete_weight_files(spec.local_dir):
        return False
    return True


def _has_complete_weight_files(path: Path) -> bool:
    weight_files = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and ".cache" not in item.parts
        and item.suffix in {".safetensors", ".bin", ".pt", ".pth"}
    ]
    if not weight_files:
        return False
    for index_path in path.rglob("*.safetensors.index.json"):
        if ".cache" in index_path.parts:
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            expected = {str(name) for name in (payload.get("weight_map") or {}).values()}
        except Exception:
            return False
        if expected and not all((index_path.parent / name).is_file() for name in expected):
            return False
    return True


def _snapshot_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [item for item in path.rglob("*") if item.is_file()]


def _result(
    spec: ModelSnapshotDownloadSpec,
    status: str,
    files: list[Path],
    *,
    error: str | None = None,
) -> ModelDownloadResult:
    return ModelDownloadResult(
        role=spec.role,
        repo_id=spec.model_id,
        local_dir=str(spec.local_dir),
        status=status,
        files=[str(path) for path in files[:50]],
        error=error,
    )
