from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


Vector = List[float]
Payload = Dict[str, Any]


@dataclass(frozen=True)
class ClipMetadata:
    clip_id: str
    path: str
    title: str = ""
    game: str = ""
    created_at: Optional[float] = None
    duration_seconds: Optional[float] = None
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    embedding: Vector
    metadata: ClipMetadata
    transcript: str = ""
    summary: str = ""
    extra: Payload = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return " ".join(
            part
            for part in [
                self.metadata.title,
                self.metadata.game,
                self.transcript,
                self.summary,
                " ".join(self.metadata.tags),
            ]
            if part
        )


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str
    confidence: float = 1.0


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    language: str = "auto"
    engine: str = "mock"
    confidence: float = 1.0


@dataclass(frozen=True)
class SummaryResult:
    title: str
    summary: str
    key_moments: List[str] = field(default_factory=list)
    engine: str = "mock"


@dataclass(frozen=True)
class TagResult:
    tags: List[str]
    confidence_by_tag: Dict[str, float] = field(default_factory=dict)
    engine: str = "rules"


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float = 1.0
    regions: List[Payload] = field(default_factory=list)
    engine: str = "mock"


@dataclass(frozen=True)
class AudioEvent:
    name: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass(frozen=True)
class AudioEventResult:
    events: List[AudioEvent] = field(default_factory=list)
    engine: str = "rules"


@dataclass(frozen=True)
class SearchRequest:
    query: str
    limit: int = 10
    game: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchResult:
    clip_id: str
    score: float
    metadata: ClipMetadata
    transcript: str = ""
    summary: str = ""
    highlights: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReasoningResult:
    answer: str
    evidence_clip_ids: List[str] = field(default_factory=list)
    engine: str = "mock"


def to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value
