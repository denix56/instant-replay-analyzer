from __future__ import annotations

import pytest

from backend.app.hf_pipeline.evidence_ledger import build_evidence_ledger, classify_death_vocalizations
from backend.app.hf_pipeline.schemas import (
    ASRSegmentV1,
    ASRTranscriptV1,
    AudioCaptionV1,
    ClipManifestV1,
    ClipTimebaseV1,
    EvidencePointerV1,
    HitMarkerEventV1,
    MetadataPayloadV1,
    VideoPayloadBudgetV1,
    VisualEventV1,
)
from backend.app.hf_pipeline.video_budget import fit_qwen_video_budget
from backend.app.processing.timebase import analysis_start_sec, analysis_to_source_time, source_to_analysis_time


def _manifest(file_name: str = "hunter_killed_clip.mp4") -> ClipManifestV1:
    metadata = MetadataPayloadV1(
        clip_id=1,
        file_name=file_name,
        file_path="/clips/hunter_killed_clip.mp4",
        user_metadata={
            "hud": {
                "equipment_timeline": [
                    {
                        "timestamp": 18.0,
                        "entity_name": "Rougarou",
                        "entity_type": "weapon",
                        "confidence": 0.95,
                    }
                ]
            },
            "hit_marker": {
                "detected": True,
                "timestamp": 19.1,
                "description": "Hit-marker detector found a cue at 19.10s.",
                "confidence": 0.9,
            },
        },
    )
    return ClipManifestV1(
        clip_id=1,
        file_name=file_name,
        file_path=metadata.file_path,
        duration_sec=25.0,
        media_type="video",
        metadata=metadata,
        ingest_timestamp="2026-05-13T00:00:00Z",
    )


def test_timebase_helpers_preserve_source_original_timestamps() -> None:
    assert analysis_start_sec(25.0, 5.0) == 5.0
    assert source_to_analysis_time(19.25, 5.0) == 14.25
    assert analysis_to_source_time(14.25, 5.0) == 19.25


def test_evidence_ledger_filters_before_skip_and_canonicalizes_rougarou() -> None:
    manifest = _manifest()
    timebase = ClipTimebaseV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        source_duration_sec=25.0,
        analysis_start_sec=5.0,
        analysis_end_sec=25.0,
    )
    transcript = ASRTranscriptV1(
        clip_id=1,
        file_name=manifest.file_name,
        model_id="whisper",
        segments=[
            ASRSegmentV1(
                clip_id=1,
                file_name=manifest.file_name,
                window_id="speech_001",
                start_sec=1.0,
                end_sec=2.0,
                model_id="whisper",
                text="before skip",
            ),
            ASRSegmentV1(
                clip_id=1,
                file_name=manifest.file_name,
                window_id="speech_002",
                start_sec=6.0,
                end_sec=7.0,
                model_id="whisper",
                text="after skip",
            ),
        ],
    )
    visual = VisualEventV1(
        clip_id=1,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=18.0,
        end_sec=20.0,
        description="The player fires from a wooden interior toward an enemy at the left opening.",
        equipment_name="Rougarou",
        evidence_pointers=[
            EvidencePointerV1(
                source="video",
                window_id="window_full_clip",
                start=18.0,
                end=20.0,
                quote_or_observation="enemy at the left opening",
            )
        ],
    )

    ledger = build_evidence_ledger(
        manifest,
        timebase=timebase,
        metadata=manifest.metadata,
        transcript=transcript,
        visual_events=[visual],
        video_payload_budgets=[VideoPayloadBudgetV1(stage="full_visual", max_frames=80, max_pixels=256000)],
    )

    assert [item.text for item in ledger.speech_segments] == ["after skip"]
    assert ledger.equipment_timeline[0].display_name == "Mosin Obrez (Rougarou skin)"
    assert ledger.visual_events[0].equipment_name == "Mosin Obrez (Rougarou skin)"
    assert ledger.known_outcome == "confirmed_hunter_kill"


def test_death_scream_near_hit_marker_links_to_player_kill_but_remote_scream_does_not() -> None:
    manifest = _manifest()
    marker = HitMarkerEventV1(
        clip_id=1,
        file_name=manifest.file_name,
        start_sec=18.9,
        end_sec=19.3,
        timestamp_sec=19.1,
        evidence_pointers=[
            EvidencePointerV1(
                source="video",
                window_id="hit_marker_detection",
                start=18.9,
                end=19.3,
                quote_or_observation="hit marker",
            )
        ],
    )
    captions = [
        AudioCaptionV1(
            clip_id=1,
            file_name=manifest.file_name,
            window_id="audio_near",
            start_sec=18.7,
            end_sec=19.2,
            model_id="midasheng",
            text="A human death scream is audible.",
        ),
        AudioCaptionV1(
            clip_id=1,
            file_name=manifest.file_name,
            window_id="audio_far",
            start_sec=10.0,
            end_sec=10.5,
            model_id="midasheng",
            text="A distant death scream is audible.",
        ),
    ]

    events = classify_death_vocalizations(
        manifest,
        captions,
        [marker],
        known_outcome="confirmed_hunter_kill",
    )

    assert [event.classification for event in events] == ["player_kill_candidate", "other_death_audio"]


def test_qwen_budget_fallback_order_and_clear_failure() -> None:
    assert fit_qwen_video_budget("full_visual").max_frames == 80
    fallback = fit_qwen_video_budget("full_visual", measured_allocated_gb=8.5, max_allocated_gb=8.0)
    assert (fallback.max_frames, fallback.max_pixels, fallback.fallback_index) == (64, 224000, 1)

    with pytest.raises(RuntimeError, match="fits under 8.00GB"):
        fit_qwen_video_budget("embedding", measured_allocated_gb=8.5, max_allocated_gb=8.0)
