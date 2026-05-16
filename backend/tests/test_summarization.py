import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.processing.summarization import ClipSummarizer
from backend.app.llm.prompts import DEEP_REASONING_SYSTEM_PROMPT, build_reasoning_prompt, build_summary_prompt
from backend.app.llm.schemas import ClipMetadata, SearchResult


def test_summarizer_describes_transcript_clip():
    result = ClipSummarizer().summarize(
        "teammate said he is on the left then I got pushed by a shotgun",
        filename="Hunt_Showdown_2026-05-11_22-10-03.mp4",
        group_name="Hunt Showdown",
        tags=["teammate comms", "shotgun", "death"],
        duration_seconds=25,
        segment_count=24,
        modality_counts={"audio_video": 24},
    )

    assert result.engine == "automatic"
    assert result.summary.startswith("Gameplay clip")
    assert "Hunt Showdown" in result.summary
    assert "transcript cues" in result.summary
    assert "shotgun" in result.summary
    assert any("Indexed into 24" in moment for moment in result.key_moments)


def test_summarizer_describes_clip_without_transcript():
    result = ClipSummarizer().summarize(
        "",
        filename="boss_lair_shotgun_fight.webm",
        group_name="Hunt Showdown",
        tags=["boss", "lair", "shotgun", "gunshots"],
        duration_seconds=25,
        segment_count=13,
        modality_counts={"video_only": 13},
    )

    assert result.engine == "automatic"
    assert "boss lair shotgun fight" in result.summary
    assert "likely involving boss, lair, shotgun and gunshots" in result.summary
    assert "indexed as 13 video-only" in result.summary
    assert any("13 video-only segments" in moment for moment in result.key_moments)


def test_summarizer_includes_audio_video_embedding_hints():
    result = ClipSummarizer().summarize(
        "teammate called out one in the window",
        filename="window_push.mp4",
        group_name="Hunt Showdown",
        tags=["teammate comms", "window"],
        av_semantic_hints=["enemy visible in window", "gunshots"],
        duration_seconds=25,
        segment_count=20,
        modality_counts={"audio_video": 20},
    )

    assert "Audio-video embeddings suggest enemy visible in window and gunshots" in result.summary
    assert any("Audio-video embedding hints: enemy visible in window, gunshots" in moment for moment in result.key_moments)


def test_summarizer_includes_position_and_movement_details():
    result = ClipSummarizer().summarize(
        "",
        filename="window_push.mp4",
        group_name="Hunt Showdown",
        tags=["indoors", "shotgun"],
        av_semantic_hints=["active weapon or item: Auto-5", "enemy visible in window", "footsteps before death"],
        duration_seconds=25,
        segment_count=12,
        modality_counts={"audio_video": 12},
    )

    assert "Position/movement:" in result.summary
    assert "window angle" in result.summary
    assert "footsteps" in result.summary


def test_summarizer_extracts_enemy_relative_direction_from_transcript():
    result = ClipSummarizer().summarize(
        "enemy on my left and one above us, I am holding the stairs",
        filename="direction_callout.mp4",
        group_name="Hunt Showdown",
        tags=["teammate comms"],
        duration_seconds=18,
        segment_count=8,
        modality_counts={"audio_video": 8},
    )

    assert "Position/movement:" in result.summary
    assert "you were holding or watching an angle" in result.summary
    assert "an enemy was to your left" in result.summary
    assert "an enemy was above you" in result.summary


def test_summarizer_extracts_teammate_evidence_only_when_present():
    result = ClipSummarizer().summarize(
        "teammate nearby is downed, teammate pushing right after the callout",
        filename="teammate_state.mp4",
        group_name="Hunt Showdown",
        tags=[],
        duration_seconds=21,
        segment_count=9,
        modality_counts={"audio_video": 9},
    )

    assert "Position/movement:" in result.summary
    assert "a teammate was nearby" in result.summary
    assert "a teammate was downed or dead" in result.summary
    assert "a teammate was pushing" in result.summary


def test_summary_prompt_requests_spatial_details_without_inventing():
    prompt = build_summary_prompt("enemy on my right", audio_events=["gunshots"])

    assert "left/right/front/behind/above" in prompt
    assert "teammate nearby/downed/pushing/comms" in prompt
    assert "Do not infer or invent" in prompt
    assert "Return valid JSON only" in prompt
    assert '"uncertainties": [string]' in prompt


def test_reasoning_prompt_requires_clip_id_citations_and_missing_evidence():
    prompt = build_reasoning_prompt(
        "what killed me?",
        [
            SearchResult(
                clip_id="12",
                metadata=ClipMetadata(clip_id="12", path="round.mp4", title="round", game="Hunt"),
                summary="Player was downed after a shot.",
                transcript="",
                score=0.9,
            )
        ],
    )

    assert "Use only the evidence above" in prompt
    assert "[clip_id=12]" in prompt
    assert "missing evidence" in prompt
    assert "Cite clip ids" in DEEP_REASONING_SYSTEM_PROMPT
