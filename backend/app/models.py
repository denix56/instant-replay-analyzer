from __future__ import annotations

try:
    from datetime import UTC, datetime
except ImportError:  # Python < 3.11 on JarvisLabs base images.
    from datetime import datetime, timezone

    UTC = timezone.utc
from pathlib import Path
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:
    class _FieldInfo:
        def __init__(self, default: Any = None, default_factory: Any | None = None) -> None:
            self.default = default
            self.default_factory = default_factory

        def value(self) -> Any:
            if self.default_factory is not None:
                return self.default_factory()
            return self.default

    def Field(default: Any = None, **kwargs: Any) -> _FieldInfo:  # type: ignore[override]
        return _FieldInfo(default=default, default_factory=kwargs.get("default_factory"))

    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **data: Any) -> None:
            annotations: dict[str, Any] = {}
            for cls in reversed(type(self).__mro__):
                annotations.update(getattr(cls, "__annotations__", {}))
            for name in annotations:
                if name in data:
                    value = data.pop(name)
                else:
                    default = getattr(type(self), name, None)
                    value = default.value() if isinstance(default, _FieldInfo) else default
                setattr(self, name, value)
            for name, value in data.items():
                setattr(self, name, value)

        def model_dump(self) -> dict[str, Any]:
            return {key: _dump_value(value) for key, value in self.__dict__.items()}

        def dict(self) -> dict[str, Any]:
            return self.model_dump()


def _dump_value(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump()
    if isinstance(value, list):
        return [_dump_value(item) for item in value]
    if isinstance(value, tuple):
        return [_dump_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump_value(item) for key, item in value.items()}
    return value

OperationStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
ClipStatus = Literal["new", "indexed", "partial", "failed", "missing", "pending"]
ScanStatus = Literal["new", "unchanged", "changed", "missing", "unknown"]
Modality = Literal["audio_video", "video_only", "audio_only", "transcript", "metadata", "hybrid"]


class ClipRecord(BaseModel):
    id: int
    file_hash: str | None = None
    filename: str
    path: str
    relative_path: str | None = None
    source_root: str | None = None
    group_name: str = "Ungrouped"
    duration: float | None = None
    size_bytes: int | None = None
    created_at: str | None = None
    modified_at: str | None = None
    indexed_at: str | None = None
    last_seen_at: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec: str | None = None
    status: str = "pending"
    scan_status: str = "unknown"
    summary: str | None = None
    error_message: str | None = None


class VideoMetadata(BaseModel):
    duration: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    has_audio: bool = False
    error_message: str | None = None


class ScanRequest(BaseModel):
    input: str | None = None
    force_verify: bool = False


class IndexRequest(BaseModel):
    input: str | None = None
    group_name: str | None = None
    force: bool = False


class ScanSummary(BaseModel):
    source_root: str
    files_seen: int = 0
    files_new: int = 0
    files_changed: int = 0
    files_unchanged: int = 0
    files_missing: int = 0
    supported_videos: int = 0
    unsupported_files: int = 0
    groups: dict[str, int] = Field(default_factory=dict)
    status: str = "completed"
    error_message: str | None = None


class SegmentRecord(BaseModel):
    id: int | None = None
    clip_id: int
    group_name: str = "Ungrouped"
    start_time: float
    end_time: float
    duration: float
    modality: str
    representative_frame_path: str | None = None
    video_segment_path: str | None = None
    audio_segment_path: str | None = None
    embedding_id: str | None = None
    embedding_model: str | None = None
    embedding_precision: str | None = None
    runtime_backend: str | None = None
    segment_settings_hash: str
    created_at: str | None = None
    error_message: str | None = None


class TranscriptSegment(BaseModel):
    clip_id: int
    start_time: float | None = None
    end_time: float | None = None
    text: str
    confidence: float | None = None
    model_name: str | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    group_name: str | None = None
    modalities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    enable_reranking: bool | None = None


class SearchResult(BaseModel):
    clip_id: int
    clip_filename: str
    source_path: str
    group_name: str
    relative_path: str | None = None
    best_timestamp: float | None = None
    segment_start: float | None = None
    segment_end: float | None = None
    preview_frame: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    score: float
    matched_modality: str = "hybrid"
    matched_reason: str = ""
    transcript_snippet: str | None = None
    active_weapon: str | None = None
    active_equipment: str | None = None
    active_equipment_type: str | None = None
    detected_loadout: list[str] = Field(default_factory=list)
    killed_by_weapon: str | None = None
    killer_name: str | None = None
    death_status: str | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    warnings: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    clip_id: int
    description: str
    important_events: list[str] = Field(default_factory=list)
    cues: list[str] = Field(default_factory=list)
    visible_text: str | None = None
    tactical_observations: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    model_name: str | None = None
    runtime_backend: str | None = None
    active_weapon: str | None = None
    active_equipment: str | None = None
    active_equipment_type: str | None = None
    detected_loadout: list[str] = Field(default_factory=list)
    killed_by_weapon: str | None = None
    killer_name: str | None = None
    death_status: str | None = None
    knowledge_facts: list[dict[str, Any]] = Field(default_factory=list)


class OperationRecord(BaseModel):
    id: str
    operation_type: str
    status: OperationStatus
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    progress_percent: float = 0.0
    current_step: str | None = None
    current_item: str | None = None
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    message: str | None = None
    errors: str | None = None


class OperationEvent(BaseModel):
    id: int | None = None
    operation_id: str
    timestamp: str
    event_type: str
    step: str | None = None
    status: str | None = None
    progress_percent: float | None = None
    current_item: str | None = None
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ServiceStatus(BaseModel):
    backend: str = "unknown"
    qdrant: str = "unknown"
    model_tier: str
    runtime_profile: str
    gpu_backend: str
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)


class GroupSummary(BaseModel):
    group_name: str
    total_videos: int = 0
    indexed_videos: int = 0
    failed_videos: int = 0
    missing_videos: int = 0
    last_indexed_at: str | None = None


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def path_to_str(path: str | Path | None) -> str | None:
    return str(path) if path is not None else None
