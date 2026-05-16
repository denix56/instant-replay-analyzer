from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0"
EvidenceSource = Literal["video", "speech", "audio", "metadata"]
EmbeddingField = Literal["video", "summary", "speech", "audio_caption", "metadata", "fused"]
DeathVocalizationClass = Literal["player_kill_candidate", "other_death_audio", "uncertain_death_audio"]
QwenVideoStage = Literal["full_visual", "focus_visual", "ocr", "embedding"]


class VersionedModel(BaseModel):
    schema_version: str = SCHEMA_VERSION


class MetadataPayloadV1(VersionedModel):
    clip_id: int | str
    file_name: str
    file_path: str | None = None
    source_uri: str | None = None
    title: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    technical_metadata: dict[str, Any] = Field(default_factory=dict)
    ingest_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("file_name")
    @classmethod
    def file_name_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("file_name is required")
        return value


class ClipManifestV1(VersionedModel):
    clip_id: int | str
    file_name: str
    file_path: str | None = None
    duration_sec: float
    media_type: str
    metadata: MetadataPayloadV1
    created_at: str | None = None
    ingest_timestamp: str | None = None

    @model_validator(mode="after")
    def validate_duration(self) -> "ClipManifestV1":
        if self.duration_sec < 0:
            raise ValueError("duration_sec must be non-negative")
        if not (self.created_at or self.ingest_timestamp):
            raise ValueError("created_at or ingest_timestamp is required")
        return self


class MediaWindowV1(VersionedModel):
    clip_id: int | str
    file_name: str
    window_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    frame_paths: list[str] = Field(default_factory=list)
    video_path: str | None = None
    prepared_video_frame_paths: list[str] = Field(default_factory=list)
    prepared_video_sample_fps: float | None = None
    prepared_video_metadata: dict[str, Any] = Field(default_factory=dict)
    audio_path: str | None = None

    @model_validator(mode="after")
    def validate_window(self) -> "MediaWindowV1":
        if self.start_sec < 0:
            raise ValueError("start_sec must be non-negative")
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be >= start_sec")
        if self.duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        return self


class EvidencePointerV1(VersionedModel):
    source: EvidenceSource
    window_id: str | None = None
    start: float
    end: float
    quote_or_observation: str

    @model_validator(mode="after")
    def validate_pointer(self) -> "EvidencePointerV1":
        if self.start < 0:
            raise ValueError("evidence pointer start must be non-negative")
        if self.end < self.start:
            raise ValueError("evidence pointer end must be >= start")
        if not self.quote_or_observation.strip():
            raise ValueError("quote_or_observation is required")
        return self


class EvidenceTextModel(VersionedModel):
    clip_id: int | str
    file_name: str
    window_id: str
    start_sec: float
    end_sec: float
    source: EvidenceSource
    model_id: str
    text: str
    confidence: float | None = None
    uncertainties: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timing_and_text(self) -> "EvidenceTextModel":
        if self.start_sec < 0:
            raise ValueError("start_sec must be non-negative")
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be >= start_sec")
        if not self.text.strip():
            raise ValueError("text is required")
        return self


class VideoObservationV1(EvidenceTextModel):
    source: Literal["video"] = "video"


class ASRSegmentV1(EvidenceTextModel):
    source: Literal["speech"] = "speech"


class ASRTranscriptV1(VersionedModel):
    clip_id: int | str
    file_name: str
    model_id: str
    language: str | None = None
    text: str = ""
    segments: list[ASRSegmentV1] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class AudioCaptionV1(EvidenceTextModel):
    source: Literal["audio"] = "audio"

    @model_validator(mode="after")
    def mark_uncertain(self) -> "AudioCaptionV1":
        if not self.uncertainties:
            self.uncertainties.append("Audio caption evidence may hallucinate environmental sounds.")
        return self


class KeyMomentV1(BaseModel):
    start: float
    end: float
    description: str
    evidence: list[EvidenceSource]
    evidence_pointers: list[EvidencePointerV1]

    @model_validator(mode="after")
    def validate_key_moment(self) -> "KeyMomentV1":
        if self.start < 0:
            raise ValueError("key moment start must be non-negative")
        if self.end < self.start:
            raise ValueError("key moment end must be >= start")
        if not self.description.strip():
            raise ValueError("key moment description is required")
        if not self.evidence:
            raise ValueError("key moment requires at least one evidence source")
        if not self.evidence_pointers:
            raise ValueError("key moment requires at least one evidence pointer")
        return self


class FusedSummaryV1(VersionedModel):
    clip_id: int | str
    file_name: str
    model_id: str
    title: str
    short_summary: str
    detailed_summary: str
    key_moments: list[KeyMomentV1]
    tags: list[str] = Field(default_factory=list)
    detected_language: str | None = None
    uncertainties: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_summary(self) -> "FusedSummaryV1":
        if not self.title.strip():
            raise ValueError("title is required")
        if not self.short_summary.strip():
            raise ValueError("short_summary is required")
        if not self.detailed_summary.strip():
            raise ValueError("detailed_summary is required")
        return self

    def validate_evidence_bounds(self, duration_sec: float) -> None:
        for moment in self.key_moments:
            if moment.end > duration_sec:
                raise ValueError(f"key moment exceeds clip duration: {moment.end} > {duration_sec}")
            pointer_sources = {pointer.source for pointer in moment.evidence_pointers}
            if not pointer_sources.issubset(set(moment.evidence)):
                raise ValueError("key moment evidence must include every evidence pointer source")
            if pointer_sources == {"audio"} and not self.uncertainties:
                raise ValueError("audio-only claims must be marked uncertain")
            for pointer in moment.evidence_pointers:
                if pointer.end > duration_sec:
                    raise ValueError(f"evidence pointer exceeds clip duration: {pointer.end} > {duration_sec}")


class ClipTimebaseV1(VersionedModel):
    clip_id: int | str
    file_name: str
    source_duration_sec: float
    analysis_start_sec: float
    analysis_end_sec: float
    timestamp_policy: Literal["source_original"] = "source_original"

    @model_validator(mode="after")
    def validate_timebase(self) -> "ClipTimebaseV1":
        if self.source_duration_sec < 0:
            raise ValueError("source_duration_sec must be non-negative")
        if self.analysis_start_sec < 0:
            raise ValueError("analysis_start_sec must be non-negative")
        if self.analysis_end_sec < self.analysis_start_sec:
            raise ValueError("analysis_end_sec must be >= analysis_start_sec")
        if self.analysis_end_sec > self.source_duration_sec + 1e-6:
            raise ValueError("analysis_end_sec must be within source duration")
        return self


class EquipmentStateV1(VersionedModel):
    clip_id: int | str
    file_name: str
    start_sec: float
    end_sec: float
    raw_name: str
    display_name: str
    equipment_type: str = "item"
    source: EvidenceSource = "video"
    confidence: float | None = None
    evidence_pointers: list[EvidencePointerV1] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_equipment(self) -> "EquipmentStateV1":
        if self.start_sec < 0:
            raise ValueError("equipment start_sec must be non-negative")
        if self.end_sec < self.start_sec:
            raise ValueError("equipment end_sec must be >= start_sec")
        if not self.display_name.strip():
            raise ValueError("equipment display_name is required")
        return self


class HitMarkerEventV1(VersionedModel):
    clip_id: int | str
    file_name: str
    start_sec: float
    end_sec: float
    timestamp_sec: float
    confidence: float | None = None
    associated_with_player_kill: bool = False
    evidence_pointers: list[EvidencePointerV1] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_hit_marker(self) -> "HitMarkerEventV1":
        if self.timestamp_sec < 0 or self.start_sec < 0:
            raise ValueError("hit-marker timestamps must be non-negative")
        if self.end_sec < self.start_sec:
            raise ValueError("hit-marker end_sec must be >= start_sec")
        if not self.evidence_pointers:
            raise ValueError("hit-marker event requires evidence pointers")
        return self


class DeathScreenEventV1(VersionedModel):
    clip_id: int | str
    file_name: str
    start_sec: float
    end_sec: float
    status: str | None = None
    killed_with: str | None = None
    killer_name: str | None = None
    confidence: float | None = None
    evidence_pointers: list[EvidencePointerV1] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_death_screen(self) -> "DeathScreenEventV1":
        if self.start_sec < 0:
            raise ValueError("death-screen start_sec must be non-negative")
        if self.end_sec < self.start_sec:
            raise ValueError("death-screen end_sec must be >= start_sec")
        if not (self.status or self.killed_with or self.killer_name or self.raw_payload):
            raise ValueError("death-screen event requires extracted content")
        return self


class DeathVocalizationEventV1(VersionedModel):
    clip_id: int | str
    file_name: str
    start_sec: float
    end_sec: float
    text: str
    classification: DeathVocalizationClass
    associated_hit_marker_time_sec: float | None = None
    confidence: float | None = None
    evidence_pointers: list[EvidencePointerV1] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_death_vocalization(self) -> "DeathVocalizationEventV1":
        if self.start_sec < 0:
            raise ValueError("death-vocalization start_sec must be non-negative")
        if self.end_sec < self.start_sec:
            raise ValueError("death-vocalization end_sec must be >= start_sec")
        if not self.text.strip():
            raise ValueError("death-vocalization text is required")
        if not self.evidence_pointers:
            raise ValueError("death-vocalization event requires evidence pointers")
        return self


class VisualEventV1(VersionedModel):
    clip_id: int | str
    file_name: str
    window_id: str
    start_sec: float
    end_sec: float
    description: str
    player_position: str | None = None
    enemy_position: str | None = None
    teammate_positions: list[str] = Field(default_factory=list)
    equipment_name: str | None = None
    action: str | None = None
    confidence: float | None = None
    evidence_pointers: list[EvidencePointerV1] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_visual_event(self) -> "VisualEventV1":
        if self.start_sec < 0:
            raise ValueError("visual event start_sec must be non-negative")
        if self.end_sec < self.start_sec:
            raise ValueError("visual event end_sec must be >= start_sec")
        if not self.description.strip():
            raise ValueError("visual event description is required")
        if not self.evidence_pointers:
            raise ValueError("visual event requires evidence pointers")
        return self


class VideoPayloadBudgetV1(VersionedModel):
    stage: QwenVideoStage
    max_frames: int
    max_pixels: int
    fallback_index: int = 0
    max_allocated_gb: float = 8.0
    measured_allocated_gb: float | None = None

    @model_validator(mode="after")
    def validate_budget(self) -> "VideoPayloadBudgetV1":
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
        if self.max_allocated_gb <= 0:
            raise ValueError("max_allocated_gb must be positive")
        if self.measured_allocated_gb is not None and self.measured_allocated_gb < 0:
            raise ValueError("measured_allocated_gb must be non-negative")
        return self


class EvidenceLedgerV1(VersionedModel):
    clip_id: int | str
    file_name: str
    timebase: ClipTimebaseV1
    metadata: MetadataPayloadV1
    equipment_timeline: list[EquipmentStateV1] = Field(default_factory=list)
    visual_events: list[VisualEventV1] = Field(default_factory=list)
    speech_segments: list[ASRSegmentV1] = Field(default_factory=list)
    audio_captions: list[AudioCaptionV1] = Field(default_factory=list)
    hit_markers: list[HitMarkerEventV1] = Field(default_factory=list)
    death_screen_events: list[DeathScreenEventV1] = Field(default_factory=list)
    death_vocalizations: list[DeathVocalizationEventV1] = Field(default_factory=list)
    video_payload_budgets: list[VideoPayloadBudgetV1] = Field(default_factory=list)
    known_outcome: str | None = None
    uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ledger(self) -> "EvidenceLedgerV1":
        if str(self.clip_id) != str(self.timebase.clip_id):
            raise ValueError("ledger clip_id must match timebase clip_id")
        duration = self.timebase.source_duration_sec
        for pointer in self.evidence_pointers():
            if pointer.end > duration + 1e-6:
                raise ValueError(f"evidence pointer exceeds clip duration: {pointer.end} > {duration}")
        for collection_name in (
            "equipment_timeline",
            "visual_events",
            "speech_segments",
            "audio_captions",
            "hit_markers",
            "death_screen_events",
            "death_vocalizations",
        ):
            for item in getattr(self, collection_name):
                start = float(getattr(item, "start_sec", 0.0))
                end = float(getattr(item, "end_sec", start))
                if start + 1e-6 < self.timebase.analysis_start_sec:
                    raise ValueError(f"{collection_name} contains evidence before analysis_start_sec")
                if end > duration + 1e-6:
                    raise ValueError(f"{collection_name} exceeds source duration")
        return self

    def evidence_pointers(self) -> list[EvidencePointerV1]:
        pointers: list[EvidencePointerV1] = []
        for collection_name in (
            "equipment_timeline",
            "visual_events",
            "hit_markers",
            "death_screen_events",
            "death_vocalizations",
        ):
            for item in getattr(self, collection_name):
                pointers.extend(getattr(item, "evidence_pointers", []))
        for item in self.speech_segments:
            pointers.append(EvidencePointerV1(source="speech", window_id=item.window_id, start=item.start_sec, end=item.end_sec, quote_or_observation=item.text))
        for item in self.audio_captions:
            pointers.append(EvidencePointerV1(source="audio", window_id=item.window_id, start=item.start_sec, end=item.end_sec, quote_or_observation=item.text))
        return pointers


class EmbeddingRecordV1(VersionedModel):
    clip_id: int | str
    file_name: str
    field: EmbeddingField
    window_id: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    model_id: str
    embedding_dim: int
    payload_hash: str
    payload_text: str | None = None
    payload_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_embedding_record(self) -> "EmbeddingRecordV1":
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not (self.payload_text or self.payload_ref):
            raise ValueError("payload_text or payload_ref is required")
        if not self.file_name.strip():
            raise ValueError("file_name is required")
        return self


class RetrievalCandidateV1(VersionedModel):
    clip_id: int | str
    file_name: str
    combined_score: float
    matched_fields: list[EmbeddingField] = Field(default_factory=list)
    field_scores: dict[str, float] = Field(default_factory=dict)
    evidence_pointers: list[EvidencePointerV1] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class RetrievalResultV1(VersionedModel):
    query: str
    candidates: list[RetrievalCandidateV1]
    per_field_counts: dict[str, int] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)


class RerankCandidateV1(VersionedModel):
    clip_id: int | str
    file_name: str
    query: str
    context: str
    matched_vector_fields: list[EmbeddingField] = Field(default_factory=list)
    retrieval_scores: dict[str, float] = Field(default_factory=dict)
    evidence_pointers: list[EvidencePointerV1] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankResultV1(VersionedModel):
    query: str
    model_id: str
    ranked_clip_ids: list[int | str]
    scores: dict[str, float]
    candidates: list[RerankCandidateV1] = Field(default_factory=list)


def payload_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
