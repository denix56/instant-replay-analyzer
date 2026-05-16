from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..embeddings.vector_store import VectorSearchHit
from .schemas import EmbeddingField, EvidencePointerV1, RetrievalCandidateV1, RetrievalResultV1


VALID_FIELDS: tuple[EmbeddingField, ...] = ("video", "summary", "speech", "audio_caption", "metadata", "fused")
DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "fused": 0.35,
    "video": 0.20,
    "summary": 0.20,
    "speech": 0.15,
    "audio_caption": 0.05,
    "metadata": 0.05,
}


@dataclass(frozen=True)
class RetrievalSettings:
    per_field_top_k: int = 50
    rerank_top_n: int = 50
    final_top_k: int = 10
    score_normalization: str = "minmax_per_field"
    field_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FIELD_WEIGHTS))


def normalize_scores(hits: list[VectorSearchHit]) -> dict[str, float]:
    if not hits:
        return {}
    scores = [float(hit.score) for hit in hits]
    low = min(scores)
    high = max(scores)
    if high == low:
        return {hit.point_id: 1.0 for hit in hits}
    return {hit.point_id: (float(hit.score) - low) / (high - low) for hit in hits}


def late_fuse_hits(
    query: str,
    hits_by_field: dict[str, list[VectorSearchHit]],
    *,
    settings: RetrievalSettings | None = None,
) -> RetrievalResultV1:
    selected_settings = settings or RetrievalSettings()
    by_clip: dict[str, dict[str, Any]] = {}
    per_field_counts: dict[str, int] = {}
    for field, hits in hits_by_field.items():
        if field not in VALID_FIELDS:
            continue
        per_field_counts[field] = len(hits)
        normalized = normalize_scores(hits)
        weight = selected_settings.field_weights.get(field, 0.0)
        for hit in hits:
            clip_id = str(hit.payload.get("clip_id"))
            if not clip_id or clip_id == "None":
                continue
            bucket = by_clip.setdefault(
                clip_id,
                {
                    "clip_id": hit.payload.get("clip_id"),
                    "file_name": str(hit.payload.get("file_name") or hit.payload.get("filename") or ""),
                    "combined_score": 0.0,
                    "matched_fields": set(),
                    "field_scores": {},
                    "evidence_pointers": [],
                    "payload": dict(hit.payload),
                },
            )
            score = normalized.get(hit.point_id, 0.0)
            weighted = score * weight
            bucket["combined_score"] += weighted
            bucket["matched_fields"].add(field)
            bucket["field_scores"][field] = max(float(bucket["field_scores"].get(field, 0.0)), weighted)
            pointer = evidence_pointer_from_payload(field, hit.payload)
            if pointer is not None:
                bucket["evidence_pointers"].append(pointer)
            if weighted >= bucket["field_scores"][field]:
                bucket["payload"].update(hit.payload)
    candidates = [
        RetrievalCandidateV1(
            clip_id=value["clip_id"],
            file_name=value["file_name"],
            combined_score=round(float(value["combined_score"]), 6),
            matched_fields=sorted(value["matched_fields"]),
            field_scores={key: round(float(score), 6) for key, score in value["field_scores"].items()},
            evidence_pointers=value["evidence_pointers"][:8],
            payload=value["payload"],
        )
        for value in by_clip.values()
    ]
    candidates.sort(key=lambda item: item.combined_score, reverse=True)
    return RetrievalResultV1(
        query=query,
        candidates=candidates[: selected_settings.rerank_top_n],
        per_field_counts=per_field_counts,
        settings={
            "per_field_top_k": selected_settings.per_field_top_k,
            "rerank_top_n": selected_settings.rerank_top_n,
            "final_top_k": selected_settings.final_top_k,
            "score_normalization": selected_settings.score_normalization,
            "field_weights": selected_settings.field_weights,
        },
    )


def evidence_pointer_from_payload(field: str, payload: dict[str, Any]) -> EvidencePointerV1 | None:
    source = {
        "video": "video",
        "summary": "metadata",
        "speech": "speech",
        "audio_caption": "audio",
        "metadata": "metadata",
        "fused": "metadata",
    }.get(field)
    if source is None:
        return None
    start = _float_or_zero(payload.get("start_sec", payload.get("start_time")))
    end = _float_or_zero(payload.get("end_sec", payload.get("end_time", start)))
    text = str(payload.get("payload_text") or payload.get("text") or payload.get("summary") or payload.get("file_name") or payload.get("filename") or "")
    if not text.strip():
        return None
    return EvidencePointerV1(
        source=source,  # type: ignore[arg-type]
        window_id=payload.get("window_id"),
        start=start,
        end=max(start, end),
        quote_or_observation=text[:500],
    )


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

