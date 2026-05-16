from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, Sequence

from .schemas import (
    ASRSegmentV1,
    ASRTranscriptV1,
    AudioCaptionV1,
    ClipManifestV1,
    ClipTimebaseV1,
    DeathScreenEventV1,
    DeathVocalizationEventV1,
    EquipmentStateV1,
    EvidenceLedgerV1,
    EvidencePointerV1,
    HitMarkerEventV1,
    MetadataPayloadV1,
    VideoPayloadBudgetV1,
    VisualEventV1,
)


DEATH_AUDIO_PATTERN = re.compile(
    r"\b(death\s+scream|death\s+cry|death\s+vocalization|pain\s+cry|human\s+cry|scream|groan|dying|downed|killed)\b",
    re.IGNORECASE,
)
PROMPT_ECHO_PATTERN = re.compile(
    r"\b(caption only|non-speech gameplay sounds|do not transcribe|do not infer|return one cautious)\b",
    re.IGNORECASE,
)
REPETITIVE_TEXT_PATTERN = re.compile(r"\b(\w{2,})\b(?:\W+\1\b){5,}", re.IGNORECASE)


def build_evidence_ledger(
    manifest: ClipManifestV1,
    *,
    timebase: ClipTimebaseV1,
    metadata: MetadataPayloadV1,
    transcript: ASRTranscriptV1 | None = None,
    audio_captions: Sequence[AudioCaptionV1] = (),
    visual_events: Sequence[VisualEventV1] = (),
    hit_marker_summary: Mapping[str, Any] | None = None,
    death_screen_summary: Mapping[str, Any] | None = None,
    video_payload_budgets: Sequence[VideoPayloadBudgetV1] = (),
    weapon_resolver: Callable[[str], str | None] | None = None,
    association_window_sec: float = 1.5,
) -> EvidenceLedgerV1:
    equipment = _equipment_timeline_from_metadata(manifest, metadata, weapon_resolver=weapon_resolver)
    hit_markers = _hit_markers_from_summary(manifest, hit_marker_summary or metadata.user_metadata.get("hit_marker"))
    death_screens = _death_screen_events_from_summary(
        manifest,
        death_screen_summary or metadata.user_metadata.get("death_screen"),
    )
    clean_speech = _filtered_speech_segments(transcript.segments if transcript is not None else (), timebase=timebase)
    clean_audio = _filtered_audio_captions(audio_captions, timebase=timebase)
    canonical_visual_events = _canonical_visual_events(
        visual_events,
        manifest=manifest,
        timebase=timebase,
        equipment_timeline=equipment,
        weapon_resolver=weapon_resolver,
    )
    known_outcome = _known_outcome(manifest, metadata)
    death_vocalizations = classify_death_vocalizations(
        manifest,
        clean_audio,
        hit_markers,
        known_outcome=known_outcome,
        association_window_sec=association_window_sec,
    )
    ledger = EvidenceLedgerV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        timebase=timebase,
        metadata=metadata,
        equipment_timeline=equipment,
        visual_events=canonical_visual_events,
        speech_segments=clean_speech,
        audio_captions=clean_audio,
        hit_markers=hit_markers,
        death_screen_events=death_screens,
        death_vocalizations=death_vocalizations,
        video_payload_budgets=list(video_payload_budgets),
        known_outcome=known_outcome,
        uncertainties=_ledger_uncertainties(clean_audio, death_vocalizations),
    )
    validate_evidence_ledger(ledger)
    return ledger


def validate_evidence_ledger(ledger: EvidenceLedgerV1) -> EvidenceLedgerV1:
    return EvidenceLedgerV1.model_validate(ledger.model_dump())


def classify_death_vocalizations(
    manifest: ClipManifestV1,
    audio_captions: Sequence[AudioCaptionV1],
    hit_markers: Sequence[HitMarkerEventV1],
    *,
    known_outcome: str | None = None,
    association_window_sec: float = 1.5,
) -> list[DeathVocalizationEventV1]:
    events: list[DeathVocalizationEventV1] = []
    marker_times = [float(item.timestamp_sec) for item in hit_markers]
    for caption in audio_captions:
        text = caption.text.strip()
        if not text or not DEATH_AUDIO_PATTERN.search(text):
            continue
        midpoint = (caption.start_sec + caption.end_sec) / 2.0
        nearby = min(marker_times, key=lambda value: abs(value - midpoint), default=None)
        if nearby is not None and abs(nearby - midpoint) <= association_window_sec and known_outcome:
            classification = "player_kill_candidate"
            associated = nearby
            uncertainties = [
                "Death vocalization is associated with the player-kill candidate because it is close to hit-marker and clip-context evidence."
            ]
        elif nearby is not None and abs(nearby - midpoint) <= association_window_sec:
            classification = "uncertain_death_audio"
            associated = nearby
            uncertainties = ["Death vocalization is near hit-marker evidence but the kill outcome is not confirmed."]
        else:
            classification = "other_death_audio"
            associated = None
            uncertainties = ["Death vocalization is not near player-hit evidence; treat it as someone dying elsewhere/off-screen/nearby."]
        pointer = EvidencePointerV1(
            source="audio",
            window_id=caption.window_id,
            start=caption.start_sec,
            end=caption.end_sec,
            quote_or_observation=caption.text,
        )
        events.append(
            DeathVocalizationEventV1(
                clip_id=manifest.clip_id,
                file_name=manifest.file_name,
                start_sec=caption.start_sec,
                end_sec=caption.end_sec,
                text=caption.text,
                classification=classification,  # type: ignore[arg-type]
                associated_hit_marker_time_sec=associated,
                confidence=caption.confidence,
                evidence_pointers=[pointer],
                uncertainties=uncertainties,
                raw_payload=caption.raw_payload,
            )
        )
    return events


def ledger_to_compact_text(ledger: EvidenceLedgerV1) -> str:
    payload = ledger.model_dump()
    payload["metadata"] = _compact_metadata(payload.get("metadata") or {})
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def visual_events_to_observations(ledger: EvidenceLedgerV1, *, model_id: str) -> list[Any]:
    from .schemas import VideoObservationV1

    observations: list[VideoObservationV1] = []
    for event in ledger.visual_events:
        observations.append(
            VideoObservationV1(
                clip_id=ledger.clip_id,
                file_name=ledger.file_name,
                window_id=event.window_id,
                start_sec=event.start_sec,
                end_sec=event.end_sec,
                model_id=model_id,
                text=event.description,
                confidence=event.confidence,
                uncertainties=event.uncertainties,
                raw_payload={"source_event": event.model_dump()},
            )
        )
    for hit_marker in ledger.hit_markers:
        observations.append(
            VideoObservationV1(
                clip_id=ledger.clip_id,
                file_name=ledger.file_name,
                window_id="hit_marker_detection",
                start_sec=hit_marker.start_sec,
                end_sec=hit_marker.end_sec,
                model_id="hit_marker_detector",
                text=hit_marker.evidence_pointers[0].quote_or_observation,
                confidence=hit_marker.confidence,
                uncertainties=hit_marker.uncertainties,
                raw_payload=hit_marker.raw_payload,
            )
        )
    return observations


def _filtered_speech_segments(segments: Sequence[ASRSegmentV1], *, timebase: ClipTimebaseV1) -> list[ASRSegmentV1]:
    output: list[ASRSegmentV1] = []
    for item in segments:
        text = item.text.strip()
        if not text or REPETITIVE_TEXT_PATTERN.search(text):
            continue
        if item.end_sec + 1e-6 < timebase.analysis_start_sec:
            continue
        output.append(item)
    return output


def _filtered_audio_captions(captions: Sequence[AudioCaptionV1], *, timebase: ClipTimebaseV1) -> list[AudioCaptionV1]:
    output: list[AudioCaptionV1] = []
    seen: set[tuple[float, float, str]] = set()
    for caption in captions:
        text = re.sub(r"\s+", " ", caption.text).strip()
        if not text or PROMPT_ECHO_PATTERN.search(text) or REPETITIVE_TEXT_PATTERN.search(text):
            continue
        if caption.end_sec + 1e-6 < timebase.analysis_start_sec:
            continue
        key = (round(caption.start_sec, 2), round(caption.end_sec, 2), text.lower())
        if key in seen:
            continue
        seen.add(key)
        output.append(caption.model_copy(update={"text": text}))
    return output


def _equipment_timeline_from_metadata(
    manifest: ClipManifestV1,
    metadata: MetadataPayloadV1,
    *,
    weapon_resolver: Callable[[str], str | None] | None,
) -> list[EquipmentStateV1]:
    rows: list[dict[str, Any]] = []
    for container_name in ("hud", "qwen_visual_ocr"):
        container = metadata.user_metadata.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for field in ("equipment_timeline", "prepared_frame_evidence", "evidence"):
            raw_items = container.get(field)
            if not isinstance(raw_items, list):
                continue
            for item in raw_items:
                if isinstance(item, Mapping):
                    rows.append({**item, "ledger_source": f"{container_name}.{field}"})
    states: list[EquipmentStateV1] = []
    seen: set[tuple[float, str]] = set()
    for item in sorted(rows, key=lambda row: float(row.get("timestamp") or row.get("start_timestamp") or 0.0)):
        raw_name = str(item.get("entity_name") or item.get("display_name") or item.get("raw_name") or "").strip()
        if not raw_name:
            continue
        start = _float_or_none(item.get("start_timestamp", item.get("timestamp"))) or _float_or_none(item.get("timestamp")) or 0.0
        end = _float_or_none(item.get("end_timestamp", item.get("timestamp"))) or start
        display = _canonical_equipment_name(raw_name, weapon_resolver=weapon_resolver)
        key = (round(float(start), 3), display.lower())
        if key in seen:
            continue
        seen.add(key)
        pointer = EvidencePointerV1(
            source="video",
            window_id="hud_loadout_detection" if str(item.get("ledger_source", "")).startswith("hud") else "qwen35_visual_ocr",
            start=max(0.0, float(start)),
            end=max(float(start), float(end)),
            quote_or_observation=f"current equipment: {display}",
        )
        states.append(
            EquipmentStateV1(
                clip_id=manifest.clip_id,
                file_name=manifest.file_name,
                start_sec=max(0.0, float(start)),
                end_sec=max(float(start), float(end)),
                raw_name=raw_name,
                display_name=display,
                equipment_type=str(item.get("entity_type") or "item"),
                source="video",
                confidence=_float_or_none(item.get("confidence")),
                evidence_pointers=[pointer],
                raw_payload=dict(item),
            )
        )
    return states


def _hit_markers_from_summary(manifest: ClipManifestV1, hit_marker: Any) -> list[HitMarkerEventV1]:
    if not isinstance(hit_marker, Mapping) or not hit_marker.get("detected"):
        return []
    timestamp = _float_or_none(hit_marker.get("timestamp"))
    if timestamp is None:
        evidence = hit_marker.get("evidence")
        if isinstance(evidence, list):
            timestamp = next((_float_or_none(item.get("timestamp")) for item in evidence if isinstance(item, Mapping)), None)
    timestamp = max(0.0, float(timestamp or 0.0))
    start = max(0.0, timestamp - 0.25)
    end = min(manifest.duration_sec, timestamp + 0.25)
    description = str(hit_marker.get("description") or f"Hit-marker detector found a probable hit cue at {timestamp:.2f}s.").strip()
    pointer = EvidencePointerV1(
        source="video",
        window_id="hit_marker_detection",
        start=start,
        end=max(start, end),
        quote_or_observation=description,
    )
    return [
        HitMarkerEventV1(
            clip_id=manifest.clip_id,
            file_name=manifest.file_name,
            start_sec=start,
            end_sec=max(start, end),
            timestamp_sec=timestamp,
            confidence=_float_or_none(hit_marker.get("confidence")),
            associated_with_player_kill=_known_outcome(manifest, manifest.metadata) is not None,
            evidence_pointers=[pointer],
            uncertainties=["Hit marker supports a probable hit cue; kill outcome needs corroboration."],
            raw_payload=dict(hit_marker),
        )
    ]


def _death_screen_events_from_summary(manifest: ClipManifestV1, summary: Any) -> list[DeathScreenEventV1]:
    if not isinstance(summary, Mapping):
        return []
    rows = summary.get("detections") if isinstance(summary.get("detections"), list) else [summary]
    output: list[DeathScreenEventV1] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        timestamp = max(0.0, float(_float_or_none(row.get("timestamp")) or 0.0))
        status = str(row.get("status") or "").strip() or None
        killed_with = str(row.get("killed_with") or "").strip() or None
        killer_name = str(row.get("killer_name") or "").strip() or None
        raw_text = str(row.get("raw_text") or row.get("raw_visible_text") or "").strip()
        if not (status or killed_with or killer_name or raw_text):
            continue
        pointer = EvidencePointerV1(
            source="video",
            window_id="death_screen_frame",
            start=timestamp,
            end=timestamp,
            quote_or_observation=raw_text or "death-screen UI detected",
        )
        output.append(
            DeathScreenEventV1(
                clip_id=manifest.clip_id,
                file_name=manifest.file_name,
                start_sec=timestamp,
                end_sec=timestamp,
                status=status,
                killed_with=killed_with,
                killer_name=killer_name,
                confidence=_float_or_none(row.get("confidence")),
                evidence_pointers=[pointer],
                uncertainties=[str(value) for value in row.get("uncertainties", []) if str(value).strip()]
                if isinstance(row.get("uncertainties"), list)
                else [],
                raw_payload=dict(row),
            )
        )
    return output


def _canonical_visual_events(
    events: Sequence[VisualEventV1],
    *,
    manifest: ClipManifestV1,
    timebase: ClipTimebaseV1,
    equipment_timeline: Sequence[EquipmentStateV1],
    weapon_resolver: Callable[[str], str | None] | None,
) -> list[VisualEventV1]:
    canonical_names = {item.display_name.lower() for item in equipment_timeline}
    output: list[VisualEventV1] = []
    for event in events:
        if event.end_sec + 1e-6 < timebase.analysis_start_sec:
            continue
        equipment_name = event.equipment_name
        uncertainties = list(event.uncertainties)
        if equipment_name:
            resolved = _canonical_equipment_name(equipment_name, weapon_resolver=weapon_resolver)
            if canonical_names and resolved.lower() not in canonical_names:
                uncertainties.append(
                    f"Visual event equipment '{equipment_name}' is not in the canonical local-player equipment timeline."
                )
                equipment_name = None
            else:
                equipment_name = resolved
        output.append(
            event.model_copy(
                update={
                    "clip_id": manifest.clip_id,
                    "file_name": manifest.file_name,
                    "equipment_name": equipment_name,
                    "uncertainties": uncertainties,
                }
            )
        )
    return output


def _known_outcome(manifest: ClipManifestV1, metadata: MetadataPayloadV1) -> str | None:
    raw_values = [
        manifest.file_name,
        metadata.user_metadata.get("known_outcome"),
        metadata.user_metadata.get("confirmed_outcome"),
        metadata.user_metadata.get("clip_context"),
    ]
    text = " ".join(str(value or "") for value in raw_values).lower()
    if any(token in text for token in ("confirmed_hunter_kill", "hunter_killed", "hunter killed", "_killed", " killed")):
        return "confirmed_hunter_kill"
    return None


def _canonical_equipment_name(name: str, *, weapon_resolver: Callable[[str], str | None] | None) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip()
    if weapon_resolver is not None:
        resolved = weapon_resolver(cleaned)
        if resolved:
            return resolved
    if cleaned.lower() == "rougarou":
        return "Mosin Obrez (Rougarou skin)"
    return cleaned


def _ledger_uncertainties(audio_captions: Sequence[AudioCaptionV1], death_vocalizations: Sequence[DeathVocalizationEventV1]) -> list[str]:
    uncertainties: list[str] = []
    if audio_captions:
        uncertainties.append("Audio captions are uncertain and may hallucinate environmental sounds.")
    if any(item.classification == "other_death_audio" for item in death_vocalizations):
        uncertainties.append("At least one death vocalization is not near player-hit evidence and must not be attributed to the player's kill.")
    return uncertainties


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    output = dict(metadata)
    technical = output.get("technical_metadata")
    if isinstance(technical, dict):
        output["technical_metadata"] = {
            key: value
            for key, value in technical.items()
            if key in {"duration", "width", "height", "fps", "codec", "timebase"}
        }
    return output


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
