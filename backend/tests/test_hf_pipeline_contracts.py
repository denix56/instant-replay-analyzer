import json
import wave
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.config import DEFAULT_QWEN_REASONING_BUDGET_TOKENS
from backend.app.hf_pipeline.adapters import (
    AudioCaptionerAdapter,
    FusionSummarizerAdapter,
    SUMMARY_ANSWER_MAX_TOKENS,
    _audio_caption_prompt,
    _authoritative_equipment_timeline,
    _ensure_deterministic_observation_key_moments,
    _parse_summary_contract,
    _summary_contract_errors,
    _summary_from_ledger_messages,
    _summary_messages,
    _summary_repair_messages,
    _summary_video_contract_errors,
    _video_observation_messages,
    build_asr_transcript,
    metadata_with_qwen_visual_ocr,
    parse_summary_json,
)
from backend.app.hf_pipeline.model_registry import model_for_role
from backend.app.hf_pipeline.schemas import (
    ASRSegmentV1,
    ASRTranscriptV1,
    AudioCaptionV1,
    ClipManifestV1,
    ClipTimebaseV1,
    EvidenceLedgerV1,
    EmbeddingRecordV1,
    EvidencePointerV1,
    FusedSummaryV1,
    KeyMomentV1,
    MediaWindowV1,
    MetadataPayloadV1,
    VideoObservationV1,
    payload_hash,
)
from backend.app.processing.qwen_video import SUMMARY_KILL_FOCUS_WINDOW_ID


def _metadata(file_name: str = "round_01.mp4") -> MetadataPayloadV1:
    return MetadataPayloadV1(
        clip_id=7,
        file_name=file_name,
        file_path="/safe/path/round_01.mp4",
        title="Uploaded clip",
        description="User supplied description",
    )


def _manifest(duration_sec: float = 10.0) -> ClipManifestV1:
    metadata = _metadata()
    return ClipManifestV1(
        clip_id=metadata.clip_id,
        file_name=metadata.file_name,
        file_path=metadata.file_path,
        duration_sec=duration_sec,
        media_type="video",
        metadata=metadata,
        ingest_timestamp="2026-05-13T00:00:00Z",
    )


def _write_wav(path: Path, *, sample_rate: int = 16000, channels: int = 1, duration_sec: float = 1.0) -> None:
    frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * frames * channels)


def test_required_artifact_models_include_schema_version() -> None:
    manifest = _manifest()
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_001",
        start_sec=0.0,
        end_sec=2.5,
        duration_sec=2.5,
        video_path="/safe/path/window_001.mp4",
    )
    video = VideoObservationV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id=window.window_id,
        start_sec=0.0,
        end_sec=2.5,
        model_id="Qwen/Qwen3.5-4B",
        text="Visible player movement near cover.",
    )
    speech = ASRSegmentV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="speech_001",
        start_sec=0.1,
        end_sec=1.0,
        model_id="openai/whisper-large-v3-turbo",
        text="rotate left",
    )
    audio = AudioCaptionV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id=window.window_id,
        start_sec=0.0,
        end_sec=2.5,
        model_id="mispeech/midashenglm-0.6b-fp32",
        text="Possible footsteps are audible.",
    )
    record = EmbeddingRecordV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        field="metadata",
        model_id="Qwen/Qwen3-VL-Embedding-2B",
        embedding_dim=2048,
        payload_hash=payload_hash("file_name: round_01.mp4"),
        payload_text="file_name: round_01.mp4",
    )

    for artifact in [manifest, window, video, speech, audio, record]:
        assert artifact.schema_version == "1.0"


def test_metadata_requires_file_name_with_extension_preserved() -> None:
    metadata = _metadata("teamfight.final.MP4")

    assert metadata.file_name == "teamfight.final.MP4"
    assert metadata.file_path == "/safe/path/round_01.mp4"
    with pytest.raises(ValidationError):
        _metadata(" ")


def test_audio_captioner_accepts_only_16khz_mono_chunks_at_or_below_30_seconds(tmp_path: Path) -> None:
    wav_path = tmp_path / "caption.wav"
    _write_wav(wav_path, sample_rate=16000, channels=1, duration_sec=0.25)
    manifest = _manifest(duration_sec=0.25)
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_001",
        start_sec=0.0,
        end_sec=0.25,
        duration_sec=0.25,
        audio_path=str(wav_path),
    )

    caption = AudioCaptionerAdapter().caption(manifest, window)

    assert caption is not None
    assert caption.model_id == "mispeech/midashenglm-0.6b-fp32"
    assert caption.raw_payload["runtime"] == "transformers"
    assert caption.raw_payload["loader"] == "transformers_causal_lm"
    assert caption.raw_payload["window_sec"] == 5.0
    assert caption.raw_payload["stride_sec"] == 2.5
    assert "uncertain" in caption.uncertainties[0]


def test_audio_captioner_rejects_non_16khz_non_mono_or_oversized_chunks(tmp_path: Path) -> None:
    exact_limit_path = tmp_path / "exact_limit.wav"
    _write_wav(exact_limit_path, sample_rate=16000, channels=1, duration_sec=30.0)
    adapter = AudioCaptionerAdapter()
    adapter.validate_audio(exact_limit_path)

    bad_rate_path = tmp_path / "bad_rate.wav"
    _write_wav(bad_rate_path, sample_rate=48000, channels=1)
    with pytest.raises(RuntimeError, match="16 kHz"):
        adapter.validate_audio(bad_rate_path)

    stereo_path = tmp_path / "stereo.wav"
    _write_wav(stereo_path, channels=2)

    with pytest.raises(RuntimeError, match="mono"):
        adapter.validate_audio(stereo_path)

    long_path = tmp_path / "long.wav"
    _write_wav(long_path, duration_sec=30.01)
    with pytest.raises(RuntimeError, match="30 seconds"):
        adapter.validate_audio(long_path)


def test_audio_captioner_skips_empty_real_caption(tmp_path: Path) -> None:
    wav_path = tmp_path / "caption.wav"
    _write_wav(wav_path, sample_rate=16000, channels=1, duration_sec=0.25)
    manifest = _manifest(duration_sec=0.25)
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_001",
        start_sec=0.0,
        end_sec=0.25,
        duration_sec=0.25,
        audio_path=str(wav_path),
    )

    class FakeManager:
        def caption_audio(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            return "   "

    caption = AudioCaptionerAdapter(manager=FakeManager(), mock_fallback=False).caption(manifest, window)  # type: ignore[arg-type]

    assert caption is None


def test_audio_captioner_returns_sorted_timestamped_window_captions(tmp_path: Path) -> None:
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    _write_wav(first_path, sample_rate=16000, channels=1, duration_sec=0.25)
    _write_wav(second_path, sample_rate=16000, channels=1, duration_sec=0.25)
    manifest = _manifest(duration_sec=5.0)
    windows = [
        MediaWindowV1(
            clip_id=manifest.clip_id,
            file_name=manifest.file_name,
            window_id="audio_caption_000001",
            start_sec=2.5,
            end_sec=2.75,
            duration_sec=0.25,
            audio_path=str(second_path),
        ),
        MediaWindowV1(
            clip_id=manifest.clip_id,
            file_name=manifest.file_name,
            window_id="audio_caption_000000",
            start_sec=0.0,
            end_sec=0.25,
            duration_sec=0.25,
            audio_path=str(first_path),
        ),
    ]

    captions = AudioCaptionerAdapter().caption_windows(manifest, windows)

    assert [caption.window_id for caption in captions] == ["audio_caption_000000", "audio_caption_000001"]
    assert [(caption.start_sec, caption.end_sec) for caption in captions] == [(0.0, 0.25), (2.5, 2.75)]
    assert all(caption.source == "audio" for caption in captions)
    assert all("MiDashengLM audio caption evidence is uncertain" in caption.uncertainties[0] for caption in captions)


def test_audio_caption_prompt_matches_midasheng_captioning_contract() -> None:
    prompt = _audio_caption_prompt("clip.mp4", 18.0, 20.0)

    assert "Caption only" in prompt
    assert "non-speech gameplay sounds" in prompt
    assert "death scream" in prompt
    assert "human pain cry" in prompt
    assert "do not transcribe it" in prompt
    assert "Do not infer" in prompt
    assert "clip.mp4" not in prompt


def test_asr_transcript_contract_preserves_language_detection_result() -> None:
    class Segment:
        start = 0.25
        end = 1.5
        text = "hola equipo"
        confidence = 0.8

    transcript = build_asr_transcript(
        _manifest(),
        model_id="openai/whisper-large-v3-turbo",
        text="hola equipo",
        language="es",
        segments=[Segment()],
    )

    assert transcript.language == "es"
    assert transcript.segments[0].source == "speech"
    assert transcript.segments[0].text == "hola equipo"


def test_qwen35_summarizer_generates_video_evidence_from_windows() -> None:
    manifest = _manifest()
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_001",
        start_sec=0.0,
        end_sec=2.5,
        duration_sec=2.5,
        frame_paths=["frame.jpg"],
        video_path="/safe/path/window_001.mp4",
    )
    adapter = FusionSummarizerAdapter(model_id="Qwen/Qwen3.5-4B")
    observations = adapter.observe_video_windows(manifest, [window])
    transcript = build_asr_transcript(
        manifest,
        model_id="openai/whisper-large-v3-turbo",
        text="",
        language=None,
        segments=[],
    )

    summary = adapter.summarize(
        manifest,
        video_observations=observations,
        media_windows=[window],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
    )

    assert observations[0].model_id == "Qwen/Qwen3.5-4B"
    assert window.video_path == "/safe/path/window_001.mp4"
    assert observations[0].raw_payload["video_input_mode"] == "qwen35_direct_video"
    assert "frame_paths" not in observations[0].raw_payload
    assert summary.model_id == "Qwen/Qwen3.5-4B"
    assert summary.raw_payload["fusion_mode"] == "mock_qwen35_video_aware_evidence_only"
    assert summary.key_moments[0].evidence == ["video"]


def test_summary_uses_hud_loadout_extraction_as_video_weapon_evidence() -> None:
    metadata = _metadata()
    metadata.user_metadata["hud"] = {
        "active_weapon": "Dolch 96",
        "active_equipment": "Dolch 96",
        "active_equipment_type": "weapon",
        "loadout": ["Dolch 96", "Knife"],
        "evidence": [
            {
                "segment_id": 42,
                "frame_path": "/safe/path/frame.jpg",
                "timestamp": 1.5,
                "slot_key": "current_ocr",
                "is_active": True,
                "entity_id": "weapon:dolch-96",
                "entity_name": "Dolch 96",
                "entity_type": "weapon",
                "confidence": 0.96,
                "matched_image_path": None,
            }
        ],
    }
    manifest = ClipManifestV1(
        clip_id=metadata.clip_id,
        file_name=metadata.file_name,
        file_path=metadata.file_path,
        duration_sec=10.0,
        media_type="video",
        metadata=metadata,
        ingest_timestamp="2026-05-13T00:00:00Z",
    )
    transcript = build_asr_transcript(
        manifest,
        model_id="openai/whisper-large-v3-turbo",
        text="",
        language=None,
        segments=[],
    )

    summary = FusionSummarizerAdapter(model_id="Qwen/Qwen3.5-4B").summarize(
        manifest,
        video_observations=[],
        transcript=transcript,
        audio_captions=[],
        metadata=metadata,
    )

    assert "Dolch 96" in summary.short_summary
    assert summary.key_moments[0].evidence == ["video"]
    pointer = summary.key_moments[0].evidence_pointers[0]
    assert pointer.source == "video"
    assert pointer.window_id == "hud_loadout_detection"
    assert pointer.start == 1.25
    assert "active weapon: Dolch 96" in pointer.quote_or_observation


def test_summary_uses_prepared_frame_equipment_timeline_as_video_evidence() -> None:
    metadata = _metadata()
    metadata.user_metadata["hud"] = {
        "active_weapon": "Mosin Obrez (Rougarou skin)",
        "active_equipment": "Mosin Obrez (Rougarou skin)",
        "active_equipment_type": "weapon",
        "loadout": ["Mosin Obrez (Rougarou skin)", "First Aid Kit"],
        "prepared_frame_evidence": [
            {
                "frame_index": 12,
                "frame_path": "/safe/path/frame_0012.png",
                "timestamp": 16.25,
                "slot_key": "current_ocr",
                "is_active": True,
                "entity_id": "weapon:mosin-obrez",
                "entity_name": "Mosin Obrez (Rougarou skin)",
                "entity_type": "weapon",
                "confidence": 0.96,
            },
            {
                "frame_index": 40,
                "frame_path": "/safe/path/frame_0040.png",
                "timestamp": 20.0,
                "slot_key": "current_ocr",
                "is_active": True,
                "entity_id": "tool:first-aid-kit",
                "entity_name": "First Aid Kit",
                "entity_type": "tool",
                "confidence": 0.92,
            },
        ],
        "equipment_timeline": [
            {
                "start_timestamp": 16.25,
                "end_timestamp": 18.5,
                "entity_name": "Mosin Obrez (Rougarou skin)",
                "entity_type": "weapon",
                "confidence": 0.96,
            },
            {
                "start_timestamp": 20.0,
                "end_timestamp": 20.0,
                "entity_name": "First Aid Kit",
                "entity_type": "tool",
                "confidence": 0.92,
            },
        ],
    }
    manifest = ClipManifestV1(
        clip_id=metadata.clip_id,
        file_name=metadata.file_name,
        file_path=metadata.file_path,
        duration_sec=25.0,
        media_type="video",
        metadata=metadata,
        ingest_timestamp="2026-05-13T00:00:00Z",
    )
    transcript = build_asr_transcript(
        manifest,
        model_id="openai/whisper-large-v3-turbo",
        text="",
        language=None,
        segments=[],
    )

    summary = FusionSummarizerAdapter(model_id="Qwen/Qwen3.5-4B").summarize(
        manifest,
        video_observations=[],
        transcript=transcript,
        audio_captions=[],
        metadata=metadata,
    )

    pointer = summary.key_moments[0].evidence_pointers[0]
    assert pointer.window_id == "hud_loadout_detection"
    assert "timestamped current equipment" in pointer.quote_or_observation
    assert "16.25-18.50s: Mosin Obrez (Rougarou skin)" in pointer.quote_or_observation
    assert "20.00s: First Aid Kit" in pointer.quote_or_observation


def test_summary_uses_hit_marker_extraction_as_video_evidence() -> None:
    metadata = _metadata()
    metadata.user_metadata["hit_marker"] = {
        "detected": True,
        "timestamp": 20.0,
        "confidence": 0.91,
        "description": (
            "Probable hit marker or impact cue detected near screen center at 20.00s while HUD/loadout evidence "
            "indicates active weapon Auto-5; confidence 0.91. This supports a hit cue, not a confirmed kill."
        ),
        "evidence": [
            {
                "frame_path": "/safe/path/frame_0040.png",
                "timestamp": 20.0,
                "confidence": 0.91,
                "marker_pixel_score": 330,
                "centered_target_score": 10513,
            }
        ],
    }
    manifest = ClipManifestV1(
        clip_id=metadata.clip_id,
        file_name=metadata.file_name,
        file_path=metadata.file_path,
        duration_sec=25.0,
        media_type="video",
        metadata=metadata,
        ingest_timestamp="2026-05-13T00:00:00Z",
    )
    transcript = build_asr_transcript(
        manifest,
        model_id="openai/whisper-large-v3-turbo",
        text="",
        language=None,
        segments=[],
    )

    summary = FusionSummarizerAdapter(model_id="Qwen/Qwen3.5-4B").summarize(
        manifest,
        video_observations=[],
        transcript=transcript,
        audio_captions=[],
        metadata=metadata,
    )

    assert "hit marker" in summary.short_summary.lower() or "hit marker" in summary.detailed_summary.lower()
    pointer = summary.key_moments[0].evidence_pointers[0]
    assert pointer.source == "video"
    assert pointer.window_id == "hit_marker_detection"
    assert pointer.start == 19.75
    assert "not a confirmed kill" in pointer.quote_or_observation


def test_qwen35_low_reasoning_passes_thinking_budget_to_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest()
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_001",
        start_sec=0.0,
        end_sec=2.0,
        duration_sec=2.0,
        video_path="/tmp/window_001.mp4",
    )
    calls: list[dict[str, object]] = []

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            calls.append(
                {
                    "model_id": spec.model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "chat_template_kwargs": chat_template_kwargs,
                    "thinking_budget_tokens": thinking_budget_tokens,
                    "stop_after_json": stop_after_json,
                }
            )
            if len(calls) == 1:
                return ""
            return "A player is visible inside a wooden room."

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="low",
    )
    observations = adapter.observe_video_windows(manifest, [window])

    assert observations[0].text == "A player is visible inside a wooden room."
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert calls[0]["max_new_tokens"] == 384
    assert calls[0]["thinking_budget_tokens"] == DEFAULT_QWEN_REASONING_BUDGET_TOKENS
    assert calls[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls[1]["thinking_budget_tokens"] is None


def test_qwen35_combined_summary_returns_visual_observation_and_uses_large_answer_budget() -> None:
    manifest = _manifest(duration_sec=4.0)
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=4.0,
        duration_sec=4.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png", "/tmp/frame_0001.png"],
        prepared_video_sample_fps=2.0,
    )
    transcript = ASRTranscriptV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        model_id="openai/whisper-large-v3-turbo",
        language="ru",
        text="держу угол",
        segments=[
            ASRSegmentV1(
                clip_id=manifest.clip_id,
                file_name=manifest.file_name,
                window_id="speech_001",
                start_sec=0.0,
                end_sec=1.0,
                model_id="openai/whisper-large-v3-turbo",
                text="держу угол",
            )
        ],
    )
    calls: list[dict[str, object]] = []

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            calls.append(
                {
                    "messages": messages,
                    "max_new_tokens": max_new_tokens,
                    "chat_template_kwargs": chat_template_kwargs,
                    "thinking_budget_tokens": thinking_budget_tokens,
                    "stop_after_json": stop_after_json,
                }
            )
            return json.dumps(
                {
                    "visual_observations": [
                        {
                            "window_id": "window_full_clip",
                            "start": 0.0,
                            "end": 4.0,
                            "text": "The hunter watches a dark interior corner with weapon raised.",
                            "uncertainties": [],
                        }
                    ],
                    "title": "clip.mp4",
                    "short_summary": "The hunter watches a dark interior corner.",
                    "detailed_summary": "The hunter watches a dark interior corner. Speech evidence is quoted verbatim: держу угол.",
                    "key_moments": [
                        {
                            "start": 0.0,
                            "end": 4.0,
                            "description": "Hunter holds a corner.",
                            "evidence": ["video", "speech"],
                            "evidence_pointers": [
                                {
                                    "source": "video",
                                    "window_id": "window_full_clip",
                                    "start": 0.0,
                                    "end": 4.0,
                                    "quote_or_observation": "The hunter watches a dark interior corner with weapon raised.",
                                },
                                {
                                    "source": "speech",
                                    "window_id": "speech_001",
                                    "start": 0.0,
                                    "end": 1.0,
                                    "quote_or_observation": "держу угол",
                                },
                            ],
                        }
                    ],
                    "tags": ["corner"],
                    "detected_language": "ru",
                    "uncertainties": [],
                }
            )

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="low",
    )
    observations, summary = adapter.summarize_with_observations(
        manifest,
        media_windows=[window],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
    )

    joined = json.dumps(calls[0]["messages"])
    assert "visual_observations is required" in joined
    assert observations[0].text == "The hunter watches a dark interior corner with weapon raised."
    assert observations[0].raw_payload["combined_summary"] is True
    assert summary.detailed_summary.endswith("держу угол.")
    assert calls[0]["max_new_tokens"] == SUMMARY_ANSWER_MAX_TOKENS
    assert calls[0]["thinking_budget_tokens"] == DEFAULT_QWEN_REASONING_BUDGET_TOKENS
    assert calls[0]["stop_after_json"] is True


def test_qwen35_combined_summary_reasoning_off_uses_one_bounded_answer_call() -> None:
    manifest = _manifest(duration_sec=4.0)
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=4.0,
        duration_sec=4.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png"],
        prepared_video_sample_fps=2.0,
    )
    transcript = ASRTranscriptV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        model_id="openai/whisper-large-v3-turbo",
        language=None,
        text="",
        segments=[],
    )
    calls: list[dict[str, object]] = []

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            calls.append(
                {
                    "max_new_tokens": max_new_tokens,
                    "chat_template_kwargs": chat_template_kwargs,
                    "thinking_budget_tokens": thinking_budget_tokens,
                    "stop_after_json": stop_after_json,
                }
            )
            return json.dumps(
                {
                    "visual_observations": [
                        {
                            "window_id": "window_full_clip",
                            "start": 0.0,
                            "end": 4.0,
                            "text": "The hunter moves through a dim interior.",
                            "uncertainties": [],
                        }
                    ],
                    "title": "clip.mp4",
                    "short_summary": "The hunter moves through a dim interior.",
                    "detailed_summary": "The hunter moves through a dim interior; enemies and teammates are not established.",
                    "key_moments": [
                        {
                            "start": 0.0,
                            "end": 4.0,
                            "description": "Hunter movement through interior.",
                            "evidence": ["video"],
                            "evidence_pointers": [
                                {
                                    "source": "video",
                                    "window_id": "window_full_clip",
                                    "start": 0.0,
                                    "end": 4.0,
                                    "quote_or_observation": "The hunter moves through a dim interior.",
                                }
                            ],
                        }
                    ],
                    "tags": ["interior"],
                    "detected_language": None,
                    "uncertainties": [],
                }
            )

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
        weapon_resolver=lambda text: "Mosin Obrez (Rougarou skin)" if "rougarou" in text.lower() else None,
    )
    observations, summary = adapter.summarize_with_observations(
        manifest,
        media_windows=[window],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
    )

    assert len(calls) == 1
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls[0]["thinking_budget_tokens"] is None
    assert calls[0]["max_new_tokens"] == SUMMARY_ANSWER_MAX_TOKENS
    assert calls[0]["stop_after_json"] is True
    assert observations[0].text == "The hunter moves through a dim interior."
    assert summary.short_summary == "The hunter moves through a dim interior."


def test_qwen35_summary_refines_with_focus_window_after_base_summary() -> None:
    manifest = _manifest(duration_sec=25.0)
    base_window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=25.0,
        duration_sec=25.0,
        prepared_video_frame_paths=["/tmp/base_0000.png"],
        prepared_video_sample_fps=6.0,
    )
    focus_window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id=SUMMARY_KILL_FOCUS_WINDOW_ID,
        start_sec=12.0,
        end_sec=24.0,
        duration_sec=12.0,
        prepared_video_frame_paths=["/tmp/focus_0000.png"],
        prepared_video_sample_fps=6.0,
    )
    transcript = ASRTranscriptV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        model_id="openai/whisper-large-v3",
        language=None,
        text="",
        segments=[],
    )
    calls: list[dict[str, object]] = []

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            joined = json.dumps(messages)
            calls.append({"messages": messages, "joined": joined, "max_new_tokens": max_new_tokens})
            if "Second-pass focus refinement" in joined:
                return json.dumps(
                    {
                        "visual_observations": [
                            {
                                "window_id": SUMMARY_KILL_FOCUS_WINDOW_ID,
                                "start": 18.0,
                                "end": 21.0,
                                "text": "The focus window shows the kill engagement in the doorway.",
                                "uncertainties": [],
                            }
                        ],
                        "title": "Refined kill clip",
                        "short_summary": "The refined pass describes the doorway kill engagement.",
                        "detailed_summary": "The intermediate summary is refined with the focus window: the player holds a doorway angle and the enemy position is at the doorway.",
                        "key_moments": [
                            {
                                "start": 18.0,
                                "end": 21.0,
                                "description": "Focused kill engagement at the doorway.",
                                "evidence": ["video"],
                                "evidence_pointers": [
                                    {
                                        "source": "video",
                                        "window_id": SUMMARY_KILL_FOCUS_WINDOW_ID,
                                        "start": 18.0,
                                        "end": 21.0,
                                        "quote_or_observation": "The focus window shows the kill engagement in the doorway.",
                                    }
                                ],
                            }
                        ],
                        "tags": ["engagement"],
                        "detected_language": None,
                        "uncertainties": [],
                    }
                )
            return json.dumps(
                {
                    "visual_observations": [
                        {
                            "window_id": "window_full_clip",
                            "start": 0.0,
                            "end": 25.0,
                            "text": "The base pass summarizes the full clip.",
                            "uncertainties": [],
                        }
                    ],
                    "title": "Base clip",
                    "short_summary": "The base pass summarizes the full clip.",
                    "detailed_summary": "The base pass gives broad context before the focused refinement.",
                    "key_moments": [
                        {
                            "start": 0.0,
                            "end": 25.0,
                            "description": "Broad full-clip context.",
                            "evidence": ["video"],
                            "evidence_pointers": [
                                {
                                    "source": "video",
                                    "window_id": "window_full_clip",
                                    "start": 0.0,
                                    "end": 25.0,
                                    "quote_or_observation": "The base pass summarizes the full clip.",
                                }
                            ],
                        }
                    ],
                    "tags": ["context"],
                    "detected_language": None,
                    "uncertainties": [],
                }
            )

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
    )
    observations, summary = adapter.summarize_with_observations(
        manifest,
        media_windows=[base_window, focus_window],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
    )

    assert len(calls) == 2
    assert "window_full_clip" in str(calls[0]["joined"])
    assert "Second-pass focus refinement" in str(calls[1]["joined"])
    assert "The base pass gives broad context" in str(calls[1]["joined"])
    assert summary.short_summary == "The refined pass describes the doorway kill engagement."
    assert any(item.window_id == SUMMARY_KILL_FOCUS_WINDOW_ID for item in observations)


def test_qwen35_visual_ocr_extracts_and_resolves_weapon_names_for_summary_payload() -> None:
    manifest = _manifest(duration_sec=25.0)
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=25.0,
        duration_sec=25.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png", "/tmp/frame_0001.png"],
        prepared_video_sample_fps=6.0,
        prepared_video_metadata={"qwen_video_frame_timestamps_sec": [18.0, 18.167]},
    )
    calls: list[dict[str, object]] = []

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            calls.append(
                {
                    "messages": messages,
                    "max_new_tokens": max_new_tokens,
                    "chat_template_kwargs": chat_template_kwargs,
                    "thinking_budget_tokens": thinking_budget_tokens,
                    "stop_after_json": stop_after_json,
                }
            )
            return json.dumps(
                {
                    "ocr_observations": [
                        {
                            "window_id": "window_full_clip",
                            "start": 18.0,
                            "end": 18.2,
                            "text": "The HUD shows a weapon with a 'Rougarou' skin visible.",
                            "source_area": "weapon_hud",
                            "raw_text": "Rougarou",
                            "resolved_equipment": [
                                {"raw_name": "Rougarou", "display_name": "Rougarou", "entity_type": "weapon"}
                            ],
                            "uncertainties": [],
                        }
                    ],
                    "equipment_timeline": [
                        {
                            "timestamp": 18.0,
                            "entity_name": "Rougarou",
                            "entity_type": "weapon",
                            "source": "qwen35_visual_ocr",
                            "confidence": None,
                        }
                    ],
                    "uncertainties": [],
                }
            )

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
        weapon_resolver=lambda text: "Mosin Obrez (Rougarou skin)" if "rougarou" in text.lower() else None,
        weapon_skin_map={"Rougarou": "Mosin Obrez (Rougarou skin)"},
    )

    qwen_ocr = adapter.extract_visual_ocr(manifest, media_windows=[window], metadata=manifest.metadata)
    metadata = metadata_with_qwen_visual_ocr(manifest.metadata, qwen_ocr)
    transcript = build_asr_transcript(manifest, model_id="openai/whisper-large-v3", text="", language=None, segments=[])
    messages = _summary_messages(
        manifest.model_copy(update={"metadata": metadata}),
        observations=[],
        transcript=transcript,
        audio_captions=[],
        metadata=metadata,
        media_windows=[window],
        weapon_resolver=lambda text: "Mosin Obrez (Rougarou skin)" if "rougarou" in text.lower() else None,
    )
    serialized_ocr_prompt = json.dumps(calls[0]["messages"])
    serialized_summary_prompt = json.dumps(messages)

    assert calls[0]["max_new_tokens"] == 1536
    assert "Read only the local player's currently selected or held weapon/tool/consumable text" in serialized_ocr_prompt
    assert "Do not extract kill feed" in serialized_ocr_prompt
    assert "weapon_name_resolution_rules" in serialized_ocr_prompt
    assert "Mosin Obrez (Rougarou skin)" in qwen_ocr["observations"][0]["text"]
    assert qwen_ocr["equipment_timeline"][0]["entity_name"] == "Mosin Obrez (Rougarou skin)"
    assert "qwen_visual_ocr" in serialized_summary_prompt
    assert "Qwen3.5" in serialized_summary_prompt
    assert "Mosin Obrez (Rougarou skin)" in serialized_summary_prompt


def test_qwen35_visual_ocr_uses_linear_100_frame_high_pixel_video_without_crops(tmp_path: Path) -> None:
    frame_paths = []
    timestamps = [round(index * 0.25, 3) for index in range(100)]
    for index, timestamp in enumerate(timestamps):
        path = tmp_path / f"frame_{index:04d}_{timestamp:.3f}.png"
        frame_paths.append(str(path))
    manifest = _manifest(duration_sec=25.0)
    manifest.metadata.user_metadata["hit_marker"] = {
        "detected": True,
        "timestamp": 18.0,
        "evidence": [{"timestamp": 18.0, "frame_path": frame_paths[5]}],
    }
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=25.0,
        duration_sec=25.0,
        prepared_video_frame_paths=frame_paths,
        prepared_video_sample_fps=6.0,
        prepared_video_metadata={
            "qwen_video_frame_timestamps_sec": timestamps,
        },
    )
    calls: list[dict[str, object]] = []

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            calls.append({"messages": messages})
            return json.dumps({"ocr_observations": [], "equipment_timeline": [], "uncertainties": []})

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
    )

    adapter.extract_visual_ocr(manifest, media_windows=[window], metadata=manifest.metadata)

    content = calls[0]["messages"][1]["content"]
    serialized = json.dumps(content)
    video_payload = next(item for item in content if item["type"] == "video")
    assert video_payload["max_frames"] == 50
    assert len(video_payload["video"]) == 50
    assert video_payload["max_pixels"] == 600000
    assert not any(item["type"] == "image" for item in content)
    assert "Focused equipment OCR crops follow" not in serialized
    assert "qwen-ocr-equipment-crops-v1" not in serialized
    assert "equal_time_50_frames_high_pixels_v1" in serialized
    assert "Auto-5" in serialized
    assert "not only shotgun" in serialized


def test_qwen35_visual_ocr_discards_non_equipment_text() -> None:
    manifest = _manifest(duration_sec=12.0)
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=12.0,
        duration_sec=12.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png"],
        prepared_video_sample_fps=2.0,
        prepared_video_metadata={"qwen_video_frame_timestamps_sec": [5.0]},
    )

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            return json.dumps(
                {
                    "ocr_observations": [
                        {
                            "window_id": "window_full_clip",
                            "start": 5.0,
                            "end": 5.2,
                            "text": "Teammate Boris",
                            "source_area": "teammate_tag",
                            "raw_text": "Boris",
                            "resolved_equipment": [],
                            "uncertainties": [],
                        }
                    ],
                    "equipment_timeline": [
                        {
                            "timestamp": 5.0,
                            "entity_name": "Determination",
                            "entity_type": "trait",
                            "source": "qwen35_visual_ocr",
                            "confidence": 0.7,
                        }
                    ],
                    "uncertainties": [],
                }
            )

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
        weapon_resolver=lambda text: None,
    )

    qwen_ocr = adapter.extract_visual_ocr(manifest, media_windows=[window], metadata=manifest.metadata)

    assert qwen_ocr["observations"] == []
    assert qwen_ocr["equipment_timeline"] == []


def test_qwen35_death_screen_extracts_structured_detection(tmp_path: Path) -> None:
    manifest = _manifest(duration_sec=25.0)
    frame_path = tmp_path / "death_frame.png"
    frame_path.write_bytes(b"placeholder")
    calls: list[dict[str, object]] = []

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            calls.append(
                {
                    "messages": messages,
                    "max_new_tokens": max_new_tokens,
                    "chat_template_kwargs": chat_template_kwargs,
                    "thinking_budget_tokens": thinking_budget_tokens,
                    "stop_after_json": stop_after_json,
                }
            )
            return json.dumps(
                {
                    "detections": [
                        {
                            "frame_id": "death_candidate_001",
                            "is_death_screen": True,
                            "status": "downed",
                            "killed_with": "Mosin-Nagant M1891",
                            "killer_name": "Felis",
                            "raw_visible_text": "YOU'RE DOWN\nKilled with\nMosin-Nagant M1891\nKilled by\nFelis",
                            "confidence": 0.87,
                            "uncertainties": [],
                        }
                    ],
                    "uncertainties": [],
                }
            )

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
    )

    result = adapter.extract_death_screen(
        manifest,
        frame_candidates=[
            {
                "frame_id": "death_candidate_001",
                "segment_id": 42,
                "frame_path": str(frame_path),
                "timestamp": 23.5,
            }
        ],
    )

    serialized_prompt = json.dumps(calls[0]["messages"])
    detection = result["detections"][0]
    assert calls[0]["max_new_tokens"] == 1024
    assert "death/down screens" in serialized_prompt
    assert "Do not infer hunter skin names" in serialized_prompt
    assert detection["segment_id"] == 42
    assert detection["status"] == "downed"
    assert detection["killed_with"] == "Mosin-Nagant M1891"
    assert detection["killer_name"] == "Felis"
    assert detection["source"] == "qwen35_death_screen"


def test_combined_summary_repairs_detector_only_visual_observations() -> None:
    manifest = _manifest(duration_sec=4.0)
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=4.0,
        duration_sec=4.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png"],
        prepared_video_sample_fps=6.0,
    )
    transcript = ASRTranscriptV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        model_id="openai/whisper-large-v3-turbo",
        language=None,
        text="",
        segments=[],
    )
    invalid = {
        "visual_observations": [
            {
                "window_id": "hit_marker_detection",
                "start": 1.0,
                "end": 1.5,
                "text": "Hit marker appears.",
                "uncertainties": [],
            }
        ],
        "title": "clip.mp4",
        "short_summary": "An enemy hunter shoots the player, causing a hit marker.",
        "detailed_summary": "The enemy hunter fires at the player and a hit marker appears.",
        "key_moments": [
            {
                "start": 1.0,
                "end": 1.5,
                "description": "Enemy shot player.",
                "evidence": ["video"],
                "evidence_pointers": [
                    {
                        "source": "video",
                        "window_id": "hit_marker_detection",
                        "start": 1.0,
                        "end": 1.5,
                        "quote_or_observation": "Hit marker appears.",
                    }
                ],
            }
        ],
        "tags": ["fight"],
        "detected_language": None,
        "uncertainties": [],
    }
    repaired = {
        "visual_observations": [
            {
                "window_id": "window_full_clip",
                "start": 0.0,
                "end": 4.0,
                "text": "The player fires at an enemy hunter in the room.",
                "uncertainties": [],
            }
        ],
        "title": "clip.mp4",
        "short_summary": "The player fires at an enemy hunter in the room.",
        "detailed_summary": "The player fires at an enemy hunter in the room; the hit marker is treated as local-player impact feedback.",
        "key_moments": [
            {
                "start": 1.0,
                "end": 2.0,
                "description": "Player fires at the enemy hunter.",
                "evidence": ["video"],
                "evidence_pointers": [
                    {
                        "source": "video",
                        "window_id": "window_full_clip",
                        "start": 1.0,
                        "end": 2.0,
                        "quote_or_observation": "The player fires at an enemy hunter in the room.",
                    }
                ],
            }
        ],
        "tags": ["fight"],
        "detected_language": None,
        "uncertainties": [],
    }
    calls: list[dict[str, object]] = []

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            calls.append({"messages": messages, "stop_after_json": stop_after_json})
            return json.dumps(invalid if len(calls) == 1 else repaired)

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
        weapon_resolver=lambda text: "Mosin Obrez (Rougarou skin)" if "rougarou" in text.lower() else None,
    )

    observations, summary = adapter.summarize_with_observations(
        manifest,
        media_windows=[window],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
    )

    assert len(calls) == 2
    assert "detector-only windows are insufficient" in json.dumps(calls[1]["messages"])
    assert observations[0].window_id == "window_full_clip"
    assert "The player fires" in summary.short_summary


def test_combined_summary_keeps_hit_marker_key_moment_when_model_omits_it() -> None:
    manifest = _manifest(duration_sec=25.0)
    manifest.metadata.user_metadata["hit_marker"] = {
        "detected": True,
        "timestamp": 18.083,
        "confidence": 0.99,
        "description": (
            "Probable hit marker or impact cue detected near screen center at 18.08s; confidence 0.99. "
            "This supports a hit cue, not a confirmed kill."
        ),
        "uncertainties": [
            "Hit-marker detection is deterministic visual evidence for a probable hit marker or impact cue; "
            "it does not confirm a kill by itself."
        ],
        "evidence": [
            {
                "frame_path": "/tmp/frame_0083.png",
                "timestamp": 18.083,
                "confidence": 0.99,
            }
        ],
    }
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=25.0,
        duration_sec=25.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png"],
        prepared_video_sample_fps=6.0,
    )
    transcript = ASRTranscriptV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        model_id="openai/whisper-large-v3",
        language=None,
        text="",
        segments=[],
    )
    raw_summary = {
        "visual_observations": [
            {
                "window_id": "window_full_clip",
                "start": 0.0,
                "end": 25.0,
                "text": "The player looks out a window from inside a wooden building.",
                "uncertainties": [],
            }
        ],
        "title": "Window view",
        "short_summary": "The player looks out a window from inside a wooden building.",
        "detailed_summary": "The player looks out a window from inside a wooden building.",
        "key_moments": [
            {
                "start": 0.0,
                "end": 2.0,
                "description": "The player looks out the window.",
                "evidence": ["video"],
                "evidence_pointers": [
                    {
                        "source": "video",
                        "window_id": "window_full_clip",
                        "start": 0.0,
                        "end": 2.0,
                        "quote_or_observation": "The player looks out a window from inside a wooden building.",
                    }
                ],
            }
        ],
        "tags": ["gameplay"],
        "detected_language": None,
        "uncertainties": [],
    }

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            return json.dumps(raw_summary)

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
    )

    observations, summary = adapter.summarize_with_observations(
        manifest,
        media_windows=[window],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
    )

    assert any(observation.window_id == "hit_marker_detection" for observation in observations)
    hit_moments = [
        moment
        for moment in summary.key_moments
        if any(pointer.window_id == "hit_marker_detection" for pointer in moment.evidence_pointers)
    ]
    assert len(hit_moments) == 1
    assert hit_moments[0].start == 17.833
    assert "not a confirmed kill" in hit_moments[0].description
    assert "18.08s" in summary.detailed_summary


def test_killed_clip_metadata_confirms_kill_with_position_and_weapon() -> None:
    metadata = _metadata("Hunt Showdown.Hunter killed.DVR.mp4")
    manifest = ClipManifestV1(
        clip_id=metadata.clip_id,
        file_name=metadata.file_name,
        file_path=metadata.file_path,
        duration_sec=25.0,
        media_type="video",
        metadata=metadata,
        ingest_timestamp="2026-05-13T00:00:00Z",
    )
    metadata.user_metadata["hit_marker"] = {
        "detected": True,
        "timestamp": 18.083,
        "confidence": 0.99,
        "description": "Probable hit marker or impact cue detected near screen center at 18.08s.",
        "evidence": [{"timestamp": 18.083, "frame_path": "/tmp/frame_0083.png"}],
    }
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=25.0,
        duration_sec=25.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png"],
        prepared_video_sample_fps=6.0,
    )
    transcript = ASRTranscriptV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        model_id="openai/whisper-large-v3",
        language=None,
        text="",
        segments=[],
    )
    audio_captions = [
        AudioCaptionV1(
            clip_id=manifest.clip_id,
            file_name=manifest.file_name,
            window_id="audio_caption_000006",
            start_sec=15.0,
            end_sec=20.0,
            model_id="mispeech/midashenglm-0.6b-fp32",
            text="A death scream and human pain cry are audible near the gunshot.",
        )
    ]
    raw_summary = {
        "visual_observations": [
            {
                "window_id": "window_full_clip",
                "start": 0.0,
                "end": 25.0,
                "text": (
                    "The player is inside a wooden room, firing through a window toward the outdoor walkway. "
                    "The player holds a Rougarou-marked revolver."
                ),
                "uncertainties": [],
            }
        ],
        "title": "Window kill",
        "short_summary": "The player fires from inside a wooden room through a window.",
        "detailed_summary": "The player holds a Rougarou-marked revolver and fires through the window.",
        "key_moments": [
            {
                "start": 18.0,
                "end": 18.2,
                "description": "The player fires through the window.",
                "evidence": ["video"],
                "evidence_pointers": [
                    {
                        "source": "video",
                        "window_id": "window_full_clip",
                        "start": 18.0,
                        "end": 18.2,
                        "quote_or_observation": "The player fires through the window.",
                    }
                ],
            }
        ],
        "tags": ["gameplay"],
        "detected_language": None,
        "uncertainties": [],
    }

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            return json.dumps(raw_summary)

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
        weapon_resolver=lambda text: "Mosin Obrez (Rougarou skin)" if "rougarou" in text.lower() else None,
    )

    _, summary = adapter.summarize_with_observations(
        manifest,
        media_windows=[window],
        transcript=transcript,
        audio_captions=audio_captions,
        metadata=metadata,
    )

    kill_moments = [
        moment
        for moment in summary.key_moments
        if "metadata-confirmed hunter kill" in moment.description.lower()
    ]
    assert len(kill_moments) == 1
    assert "Mosin Obrez (Rougarou skin)" in kill_moments[0].description
    assert "Rougarou-marked revolver" not in kill_moments[0].description
    assert "Mosin Obrez (Rougarou skin)" in summary.short_summary
    assert "Mosin Obrez (Rougarou skin)" in summary.detailed_summary
    assert "revolver" not in summary.short_summary.lower()
    assert "revolver" not in summary.detailed_summary.lower()
    assert "inside the wooden/windowed room" in kill_moments[0].description
    assert set(kill_moments[0].evidence) == {"video", "audio", "metadata"}
    assert "confirmed-kill" in summary.tags
    assert "clip metadata confirms a hunter kill" in summary.short_summary


def test_killed_clip_prefers_qwen_ocr_weapon_active_near_hit_marker() -> None:
    metadata = _metadata("Hunt Showdown.Hunter killed.DVR.mp4")
    metadata.user_metadata["hit_marker"] = {
        "detected": True,
        "timestamp": 18.083,
        "confidence": 0.99,
        "description": "Probable hit marker or impact cue detected near screen center at 18.08s.",
        "evidence": [{"timestamp": 18.083, "frame_path": "/tmp/frame_0083.png"}],
    }
    metadata.user_metadata["qwen_visual_ocr"] = {
        "schema_version": "1.0",
        "source": "qwen35_visual_ocr",
        "model_id": "Qwen/Qwen3.5-4B",
        "observations": [],
        "equipment_timeline": [
            {
                "timestamp": 0.0,
                "start_timestamp": 0.0,
                "end_timestamp": 5.4,
                "entity_name": "Mosin Obrez (Rougarou skin)",
                "entity_type": "weapon",
                "source": "qwen35_visual_ocr",
            },
            {
                "timestamp": 15.0,
                "start_timestamp": 15.0,
                "end_timestamp": 20.3,
                "entity_name": "Auto-5",
                "entity_type": "weapon",
                "source": "qwen35_visual_ocr",
            },
        ],
    }
    manifest = ClipManifestV1(
        clip_id=metadata.clip_id,
        file_name=metadata.file_name,
        file_path=metadata.file_path,
        duration_sec=25.0,
        media_type="video",
        metadata=metadata,
        ingest_timestamp="2026-05-13T00:00:00Z",
    )
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=25.0,
        duration_sec=25.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png"],
        prepared_video_sample_fps=6.0,
    )
    transcript = build_asr_transcript(manifest, model_id="openai/whisper-large-v3", text="", language=None, segments=[])
    raw_summary = {
        "visual_observations": [
            {
                "window_id": "window_full_clip",
                "start": 0.0,
                "end": 20.3,
                "text": "The player starts with a Mosin Obrez (Rougarou skin), later switches to Auto-5, and sees an enemy near the window.",
                "uncertainties": [],
            }
        ],
        "title": "Window kill",
        "short_summary": "The player switches from Mosin Obrez (Rougarou skin) to Auto-5 before the hit marker.",
        "detailed_summary": "The player starts with a Mosin Obrez (Rougarou skin), then has the Auto-5 active near 18s while aiming at an enemy near the window.",
        "key_moments": [
            {
                "start": 18.0,
                "end": 18.2,
                "description": "The player aims at an enemy near the window as a hit marker appears.",
                "evidence": ["video"],
                "evidence_pointers": [
                    {
                        "source": "video",
                        "window_id": "window_full_clip",
                        "start": 18.0,
                        "end": 18.2,
                        "quote_or_observation": "The player has Auto-5 active and aims at an enemy near the window.",
                    }
                ],
            }
        ],
        "tags": ["gameplay"],
        "detected_language": None,
        "uncertainties": [],
    }

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            return json.dumps(raw_summary)

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
        weapon_resolver=lambda text: "Mosin Obrez (Rougarou skin)" if "rougarou" in text.lower() else None,
    )

    _, summary = adapter.summarize_with_observations(
        manifest,
        media_windows=[window],
        transcript=transcript,
        audio_captions=[],
        metadata=metadata,
    )

    kill_moment = next(moment for moment in summary.key_moments if "metadata-confirmed hunter kill" in moment.description.lower())
    assert "with the Auto-5" in kill_moment.description
    assert "with the Auto-5" in summary.short_summary
    assert "Metadata-confirmed hunter kill" in summary.detailed_summary


def test_killed_clip_ignores_future_qwen_ocr_weapon_for_hit_marker() -> None:
    metadata = _metadata("Hunt Showdown.Hunter killed.DVR.mp4")
    metadata.user_metadata["hit_marker"] = {
        "detected": True,
        "timestamp": 18.083,
        "confidence": 0.99,
        "description": "Probable hit marker or impact cue detected near screen center at 18.08s.",
        "evidence": [{"timestamp": 18.083, "frame_path": "/tmp/frame_0083.png"}],
    }
    metadata.user_metadata["qwen_visual_ocr"] = {
        "schema_version": "1.0",
        "source": "qwen35_visual_ocr",
        "model_id": "Qwen/Qwen3.5-4B",
        "observations": [],
        "equipment_timeline": [
            {
                "timestamp": 20.4,
                "start_timestamp": 20.4,
                "end_timestamp": 20.4,
                "entity_name": "New Army",
                "entity_type": "weapon",
                "source": "qwen35_visual_ocr",
            },
            {
                "timestamp": 7.4,
                "entity_name": "Player",
                "entity_type": "player_status",
                "source": "qwen35_visual_ocr",
            },
        ],
    }
    manifest = ClipManifestV1(
        clip_id=metadata.clip_id,
        file_name=metadata.file_name,
        file_path=metadata.file_path,
        duration_sec=25.0,
        media_type="video",
        metadata=metadata,
        ingest_timestamp="2026-05-13T00:00:00Z",
    )
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=25.0,
        duration_sec=25.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png"],
        prepared_video_sample_fps=6.0,
    )
    transcript = build_asr_transcript(manifest, model_id="openai/whisper-large-v3", text="", language=None, segments=[])
    raw_summary = {
        "visual_observations": [
            {
                "window_id": "window_full_clip",
                "start": 15.0,
                "end": 20.3,
                "text": "At 18.0s, the player fires the Auto-5 shotgun at an enemy hunter near the shelves.",
                "uncertainties": [],
            }
        ],
        "title": "Window kill",
        "short_summary": "The player moves through the room, then fires the Auto-5 revolver at the enemy.",
        "detailed_summary": "The player moves through the room, then switches to the Auto-5 revolver. The enemy is near the shelves.",
        "key_moments": [
            {
                "start": 18.0,
                "end": 18.2,
                "description": "The player fires the Auto-5 revolver.",
                "evidence": ["video"],
                "evidence_pointers": [
                    {
                        "source": "video",
                        "window_id": "window_full_clip",
                        "start": 18.0,
                        "end": 18.2,
                        "quote_or_observation": "The player fires the Auto-5 revolver at an enemy hunter.",
                    }
                ],
            }
        ],
        "tags": ["gameplay"],
        "detected_language": None,
        "uncertainties": [],
    }

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            return json.dumps(raw_summary)

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
    )

    _, summary = adapter.summarize_with_observations(
        manifest,
        media_windows=[window],
        transcript=transcript,
        audio_captions=[],
        metadata=metadata,
    )

    kill_moment = next(moment for moment in summary.key_moments if "metadata-confirmed hunter kill" in moment.description.lower())
    assert "with the Auto-5" in kill_moment.description
    assert "New Army" not in kill_moment.description
    assert "Devil's Salve" not in kill_moment.description
    public_text = " ".join(
        [
            summary.short_summary,
            summary.detailed_summary,
            " ".join(moment.description for moment in summary.key_moments),
            " ".join(pointer.quote_or_observation for moment in summary.key_moments for pointer in moment.evidence_pointers),
        ]
    )
    assert "Auto-5 revolver" not in public_text


def test_summary_contract_rejects_stop_bleeding_as_downed_claim() -> None:
    manifest = _manifest(duration_sec=10.0)
    summary = FusedSummaryV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        model_id="Qwen/Qwen3.5-4B",
        title="clip",
        short_summary="The player is downed when the Stopping Bleeding prompt appears.",
        detailed_summary="The Stopping Bleeding prompt appears, so the player is downed and unable to move.",
        key_moments=[
            KeyMomentV1(
                start=5.0,
                end=6.0,
                description="Stopping Bleeding prompt appears.",
                evidence=["video"],
                evidence_pointers=[
                    EvidencePointerV1(
                        source="video",
                        window_id="window_full_clip",
                        start=5.0,
                        end=6.0,
                        quote_or_observation="Stopping Bleeding prompt appears.",
                    )
                ],
            )
        ],
        tags=[],
        detected_language=None,
        uncertainties=[],
        raw_payload={
            "visual_observations": [
                {
                    "window_id": "window_full_clip",
                    "start": 5.0,
                    "end": 6.0,
                    "text": "Stopping Bleeding prompt appears.",
                }
            ]
        },
    )
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=10.0,
        duration_sec=10.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png"],
    )

    errors = _summary_video_contract_errors(summary, media_windows=[window])

    assert any("Stop Bleeding" in error for error in errors)


def test_summary_strips_visual_hunter_skin_names_but_keeps_weapon_skin_resolution() -> None:
    manifest = _manifest(duration_sec=10.0)
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=10.0,
        duration_sec=10.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png"],
        prepared_video_sample_fps=6.0,
    )
    transcript = ASRTranscriptV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        model_id="openai/whisper-large-v3",
        language=None,
        text="",
        segments=[],
    )
    raw_summary = {
        "visual_observations": [
            {
                "window_id": "window_full_clip",
                "start": 0.0,
                "end": 10.0,
                "text": (
                    "A teammate, identified as Rougarou by the name tag, stands by the window. "
                    "Two teammates, both Rougarou, move in the room. "
                    "The player is holding a Rougarou and also holds a Rougarou-marked revolver."
                ),
                "uncertainties": [],
            }
        ],
        "title": "Window hold",
        "short_summary": (
            "The player watches two teammates, both Rougarou, in a windowed room while holding a Rougarou."
        ),
        "detailed_summary": (
            "A teammate (Rougarou) stands by the window. Another teammate, identified as Rougarou by the name tag, "
            "moves behind the player. The player is holding a Rougarou, holds a Rougarou-marked revolver, "
            "has a weapon with a 'Rougarou' skin visible on the HUD, and carries a Mosin Obrez rifle with a Rougarou skin."
        ),
        "key_moments": [
            {
                "start": 0.0,
                "end": 10.0,
                "description": "A teammate (Rougarou) stands by the window while the player holds a Rougarou-marked revolver.",
                "evidence": ["video"],
                "evidence_pointers": [
                    {
                        "source": "video",
                        "window_id": "window_full_clip",
                        "start": 0.0,
                        "end": 10.0,
                        "quote_or_observation": "A teammate, identified as Rougarou by the name tag, stands near the player.",
                    }
                ],
            }
        ],
        "tags": ["rougarou", "gameplay"],
        "detected_language": None,
        "uncertainties": [],
    }

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            return json.dumps(raw_summary)

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
        weapon_resolver=lambda text: "Mosin Obrez (Rougarou skin)" if "rougarou" in text.lower() else None,
    )

    observations, summary = adapter.summarize_with_observations(
        manifest,
        media_windows=[window],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
    )

    public_text = " ".join(
        [
            summary.short_summary,
            summary.detailed_summary,
            " ".join(moment.description for moment in summary.key_moments),
            " ".join(pointer.quote_or_observation for moment in summary.key_moments for pointer in moment.evidence_pointers),
            " ".join(observation.text for observation in observations),
        ]
    )
    assert "Mosin Obrez (Rougarou skin)" in public_text
    assert "Rougarou-marked revolver" not in public_text
    assert "weapon with a 'Rougarou' skin" not in public_text
    assert "Mosin Obrez rifle with a Rougarou skin" not in public_text
    assert "holding a Rougarou" not in public_text
    assert "holding the Mosin Obrez (Rougarou skin)" in public_text
    assert "Auto-5 Mosin Obrez" not in public_text
    assert "teammates, both Mosin" not in public_text
    assert "identified as Rougarou" not in public_text
    assert "teammate (Rougarou)" not in public_text
    assert "both Rougarou" not in public_text
    assert "rougarou" not in summary.tags
    assert "mosin-obrez" in summary.tags


def test_qwen35_combined_summary_retries_empty_response_with_no_think_json_instruction() -> None:
    manifest = _manifest(duration_sec=4.0)
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=4.0,
        duration_sec=4.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png"],
        prepared_video_sample_fps=2.0,
    )
    transcript = ASRTranscriptV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        model_id="openai/whisper-large-v3-turbo",
        language=None,
        text="",
        segments=[],
    )
    calls: list[dict[str, object]] = []

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            calls.append(
                {
                    "messages": messages,
                    "chat_template_kwargs": chat_template_kwargs,
                    "thinking_budget_tokens": thinking_budget_tokens,
                    "stop_after_json": stop_after_json,
                }
            )
            if len(calls) == 1:
                return ""
            return json.dumps(
                {
                    "visual_observations": [
                        {
                            "window_id": "window_full_clip",
                            "start": 0.0,
                            "end": 4.0,
                            "text": "The hunter is inside a dim room; enemies are not visible.",
                            "uncertainties": [],
                        }
                    ],
                    "title": "clip.mp4",
                    "short_summary": "The hunter is inside a dim room; enemies are not visible.",
                    "detailed_summary": "The hunter is inside a dim room; enemies and teammates are not established.",
                    "key_moments": [
                        {
                            "start": 0.0,
                            "end": 4.0,
                            "description": "Hunter in dim room.",
                            "evidence": ["video"],
                            "evidence_pointers": [
                                {
                                    "source": "video",
                                    "window_id": "window_full_clip",
                                    "start": 0.0,
                                    "end": 4.0,
                                    "quote_or_observation": "The hunter is inside a dim room; enemies are not visible.",
                                }
                            ],
                        }
                    ],
                    "tags": ["interior"],
                    "detected_language": None,
                    "uncertainties": [],
                }
            )

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
    )
    observations, summary = adapter.summarize_with_observations(
        manifest,
        media_windows=[window],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
    )

    assert len(calls) == 2
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls[1]["thinking_budget_tokens"] is None
    assert calls[1]["stop_after_json"] is True
    assert "previous model response was empty" in json.dumps(calls[1]["messages"])
    assert "/no_think" in json.dumps(calls[1]["messages"])
    assert observations[0].text == "The hunter is inside a dim room; enemies are not visible."
    assert summary.short_summary == "The hunter is inside a dim room; enemies are not visible."


def test_qwen35_combined_summary_repairs_repeated_generation_loop() -> None:
    manifest = _manifest(duration_sec=25.0)
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=25.0,
        duration_sec=25.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png"],
        prepared_video_sample_fps=6.0,
    )
    transcript = ASRTranscriptV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        model_id="openai/whisper-large-v3",
        language=None,
        text="",
        segments=[],
    )
    repeated = {
        "visual_observations": [
            {
                "window_id": "window_full_clip",
                "start": 0.0,
                "end": 25.0,
                "text": "The player reloads the Auto-5 at 20.0s. The player reloads the Auto-5 at 20.0s.",
                "uncertainties": [],
            }
        ],
        "title": "clip.mp4",
        "short_summary": "The player moves through an interior and engages near a doorway.",
        "detailed_summary": (
            "The player reloads the Auto-5 at 20.0s. "
            "The player reloads the Auto-5 at 20.0s. "
            "The player reloads the Auto-5 at 20.0s."
        ),
        "key_moments": [
            {
                "start": 18.0,
                "end": 19.0,
                "description": "The player engages near the doorway.",
                "evidence": ["video"],
                "evidence_pointers": [
                    {
                        "source": "video",
                        "window_id": "window_full_clip",
                        "start": 18.0,
                        "end": 19.0,
                        "quote_or_observation": "The player engages near the doorway.",
                    }
                ],
            }
        ],
        "tags": ["engagement"],
        "detected_language": None,
        "uncertainties": [],
    }
    repaired = {
        "visual_observations": [
            {
                "window_id": "window_full_clip",
                "start": 18.0,
                "end": 21.0,
                "text": "The player stays in cover and handles the Auto-5 during the doorway engagement.",
                "uncertainties": [],
            }
        ],
        "title": "clip.mp4",
        "short_summary": "The player moves through an interior and engages near a doorway.",
        "detailed_summary": "The player stays in cover, handles the Auto-5 during the doorway engagement, and avoids frame-by-frame reload narration.",
        "key_moments": [
            {
                "start": 18.0,
                "end": 21.0,
                "description": "Doorway engagement with Auto-5 handling.",
                "evidence": ["video"],
                "evidence_pointers": [
                    {
                        "source": "video",
                        "window_id": "window_full_clip",
                        "start": 18.0,
                        "end": 21.0,
                        "quote_or_observation": "The player handles the Auto-5 during the doorway engagement.",
                    }
                ],
            }
        ],
        "tags": ["engagement"],
        "detected_language": None,
        "uncertainties": [],
    }
    calls: list[dict[str, object]] = []

    class FakeManager:
        def generate_chat(  # noqa: ANN001, ANN201
            self,
            spec,
            messages,
            *,
            temperature,
            max_new_tokens,
            chat_template_kwargs,
            thinking_budget_tokens=None,
            stop_after_json=False,
        ):
            calls.append({"messages": messages, "max_new_tokens": max_new_tokens, "stop_after_json": stop_after_json})
            return json.dumps(repeated if len(calls) == 1 else repaired)

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
        reasoning_mode="off",
    )

    observations, summary = adapter.summarize_with_observations(
        manifest,
        media_windows=[window],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
    )

    assert len(calls) == 2
    assert calls[0]["max_new_tokens"] == SUMMARY_ANSWER_MAX_TOKENS
    assert calls[1]["stop_after_json"] is True
    assert all(message["role"] != "assistant" for message in calls[1]["messages"])
    assert "The invalid response is intentionally omitted" in json.dumps(calls[1]["messages"])
    assert "repeats" in json.dumps(calls[1]["messages"])
    assert "Remove repeated or near-duplicate sentences" in json.dumps(calls[1]["messages"])
    assert observations[0].text == "The player stays in cover and handles the Auto-5 during the doorway engagement."
    assert "frame-by-frame reload narration" in summary.detailed_summary


def test_summary_repair_messages_do_not_replay_invalid_assistant_text() -> None:
    messages = [{"role": "user", "content": "Return the summary JSON."}]
    repaired_messages = _summary_repair_messages(
        messages,
        raw='{"detailed_summary":"Devil\'s Salve Devil\'s Salve","key_moments":[]}',
        error="invalid evidence source: focus",
    )
    serialized = json.dumps(repaired_messages)

    assert all(message["role"] != "assistant" for message in repaired_messages)
    assert "Devil's Salve Devil's Salve" not in serialized
    assert "Allowed evidence source labels are exactly video, speech, audio, metadata" in serialized
    assert "Never use focus as an evidence source label" in serialized


def test_summary_contract_rejects_too_many_visual_observations() -> None:
    manifest = _manifest(duration_sec=10.0)
    summary = FusedSummaryV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        model_id="Qwen/Qwen3.5-4B",
        title="clip.mp4",
        short_summary="The player moves through cover.",
        detailed_summary="The player moves through cover while enemies are not established.",
        key_moments=[
            KeyMomentV1(
                start=0.0,
                end=1.0,
                description="Movement through cover.",
                evidence=["video"],
                evidence_pointers=[
                    EvidencePointerV1(
                        source="video",
                        window_id="window_full_clip",
                        start=0.0,
                        end=1.0,
                        quote_or_observation="The player moves through cover.",
                    )
                ],
            )
        ],
        tags=[],
        raw_payload={
            "visual_observations": [
                {
                    "window_id": "window_full_clip",
                    "start": float(index),
                    "end": float(index) + 0.5,
                    "text": f"Unique visual event {index}.",
                    "uncertainties": [],
                }
                for index in range(9)
            ]
        },
    )

    errors = _summary_contract_errors(summary, media_windows=[], metadata=None, weapon_resolver=None)

    assert any("visual_observations contains too many items" in error for error in errors)


def test_qwen35_video_observation_does_not_call_model_without_video_path() -> None:
    manifest = _manifest()
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_001",
        start_sec=0.0,
        end_sec=2.0,
        duration_sec=2.0,
        frame_paths=["/tmp/representative.jpg"],
    )

    class FakeManager:
        def generate_chat(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            raise AssertionError("representative frames must not be used as model evidence")

    adapter = FusionSummarizerAdapter(
        model_id="Qwen/Qwen3.5-4B",
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
    )
    observations = adapter.observe_video_windows(manifest, [window])

    assert observations[0].raw_payload["video_input_mode"] == "no_video_input"
    assert "frame_paths" not in observations[0].raw_payload
    assert observations[0].uncertainties


def test_video_observation_prompt_is_conservative_and_plain_text() -> None:
    manifest = _manifest()
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_001",
        start_sec=0.0,
        end_sec=2.5,
        duration_sec=2.5,
        frame_paths=[],
    )

    messages = _video_observation_messages(manifest, window)
    joined = json.dumps(messages)

    assert messages[0]["role"] == "system"
    assert "file name as an identifier" in joined
    assert "Do not invent names" in joined
    assert "Describe the clip chronologically and spatially" in joined
    assert "visible enemy positions" in joined
    assert "visible teammate positions" in joined
    assert "80-180 words" in joined
    assert "No markdown and no JSON" in joined


def test_video_observation_prompt_uses_qwen35_default_video_bounds() -> None:
    manifest = _manifest()
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_001",
        start_sec=0.0,
        end_sec=2.5,
        duration_sec=2.5,
        frame_paths=[],
        video_path="/tmp/clip.mp4",
    )
    spec = model_for_role("summarizer", "default", device_backend="cuda")

    messages = _video_observation_messages(manifest, window, spec)
    video_payload = messages[1]["content"][1]

    assert video_payload["fps"] == 6.0
    assert video_payload["max_frames"] == 80
    assert video_payload["max_pixels"] == 256000


def test_text_only_ledger_composer_does_not_attach_video_payloads() -> None:
    manifest = _manifest(duration_sec=25.0)
    timebase = ClipTimebaseV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        source_duration_sec=25.0,
        analysis_start_sec=5.0,
        analysis_end_sec=25.0,
    )
    ledger = EvidenceLedgerV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        timebase=timebase,
        metadata=manifest.metadata,
        known_outcome="confirmed_hunter_kill",
    )

    messages = _summary_from_ledger_messages(manifest, ledger=ledger)
    content = messages[1]["content"]

    assert isinstance(content, str)
    assert '"analysis_start_sec": 5.0' in content
    assert '"known_outcome": "confirmed_hunter_kill"' in content
    assert '"type": "video"' not in content
    assert '"type": "image"' not in content


def test_video_observation_prompt_uses_prepared_video_frames(tmp_path: Path) -> None:
    manifest = _manifest()
    frames = []
    for index in range(2):
        frame = tmp_path / f"prepared_{index}.png"
        frame.write_bytes(b"fake")
        frames.append(str(frame))
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=10.0,
        duration_sec=10.0,
        prepared_video_frame_paths=frames,
        prepared_video_sample_fps=2.0,
    )
    spec = model_for_role("summarizer", "default", device_backend="cuda")

    messages = _video_observation_messages(manifest, window, spec)
    video_payload = messages[1]["content"][1]

    assert video_payload["type"] == "video"
    assert video_payload["video"] == frames
    assert video_payload["sample_fps"] == 2.0
    assert video_payload["max_frames"] == 2


def test_video_observation_prompt_uses_compact_video_bounds_on_metal() -> None:
    manifest = _manifest()
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_001",
        start_sec=0.0,
        end_sec=2.5,
        duration_sec=2.5,
        frame_paths=[],
        video_path="/tmp/clip.mp4",
    )
    spec = model_for_role("summarizer", "default", device_backend="macos-metal")

    messages = _video_observation_messages(manifest, window, spec)
    video_payload = messages[1]["content"][1]

    assert video_payload["fps"] == 6.0
    assert video_payload["min_frames"] == 4
    assert video_payload["max_frames"] == 80
    assert video_payload["max_pixels"] == 256000


def test_video_observation_prompt_does_not_use_representative_frames(tmp_path: Path) -> None:
    manifest = _manifest()
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake")
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_001",
        start_sec=0.0,
        end_sec=2.5,
        duration_sec=2.5,
        frame_paths=[str(frame)],
    )
    spec = model_for_role("summarizer", "default", device_backend="macos-metal")

    messages = _video_observation_messages(manifest, window, spec)

    assert len(messages[1]["content"]) == 1
    assert "frame" not in str(messages[1]["content"]).lower()


def test_summary_prompt_requires_json_evidence_and_uncertainty() -> None:
    manifest = _manifest()
    transcript = build_asr_transcript(
        manifest,
        model_id="openai/whisper-large-v3-turbo",
        text="",
        language=None,
        segments=[],
    )

    messages = _summary_messages(
        manifest,
        observations=[],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
    )
    joined = json.dumps(messages)

    assert "Return valid JSON only" in joined
    assert "no code fences" in joined
    assert "Every key moment" in joined
    assert "file_name as metadata only" in joined
    assert "HUD/loadout observations are extracted from prepared Qwen frames or representative video frames" in joined
    assert "include that equipment in short_summary or detailed_summary" in joined
    assert "latest row at or before a timestamp" in joined
    assert "Do not claim the player switched back to a previous weapon" in joined
    assert "belongs to the local first-person player/hunter" in joined
    assert "Use explicit actors" in joined
    assert "detailed_summary must reconstruct the clip in detail" in joined
    assert "player/hunter position and movement" in joined
    assert "where enemies and teammates are visible or unknown" in joined
    assert "Speech evidence must stay verbatim" in joined
    assert "Preserve it verbatim in speech evidence pointers" in joined
    assert "within 0 and 10.000" in joined


def test_summary_prompt_includes_weapon_resolution_evidence_with_timestamps() -> None:
    manifest = _manifest(duration_sec=25.0)
    manifest.metadata.user_metadata["hud"] = {
        "prepared_frame_evidence": [
            {
                "frame_index": 99,
                "frame_path": "/safe/path/frame_0099.png",
                "timestamp": 18.083,
                "slot_key": "current_ocr",
                "is_active": True,
                "entity_name": "Rougarou",
                "entity_type": "weapon",
                "confidence": 0.96,
            }
        ],
        "equipment_timeline": [
            {
                "timestamp": 18.083,
                "start_timestamp": 18.083,
                "end_timestamp": 18.083,
                "entity_name": "Rougarou",
                "entity_type": "weapon",
                "confidence": 0.96,
            }
        ],
    }
    transcript = build_asr_transcript(
        manifest,
        model_id="openai/whisper-large-v3-turbo",
        text="",
        language=None,
        segments=[],
    )

    messages = _summary_messages(
        manifest,
        observations=[],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
        weapon_resolver=lambda text: "Mosin Obrez (Rougarou skin)" if "Rougarou" in text else None,
    )
    serialized = json.dumps(messages)

    timeline = _authoritative_equipment_timeline(
        manifest.metadata,
        weapon_resolver=lambda text: "Mosin Obrez (Rougarou skin)" if "Rougarou" in text else None,
    )
    assert timeline[0]["display_name"] == "Mosin Obrez (Rougarou skin)"
    assert "weapon_resolution_evidence" in serialized
    assert "authoritative_equipment_timeline" in serialized
    assert "Rougarou" in serialized
    assert "Mosin Obrez (Rougarou skin)" in serialized
    assert "18.083" in serialized
    assert "weapon/equipment name only" in serialized
    assert "write exactly that supplied display_name" in serialized
    assert "Do not append, prepend, or substitute an item type or weapon class" in serialized
    assert "use only the supplied display_name as the equipment name" in serialized
    assert "A contradiction with authoritative_equipment_timeline is invalid" in serialized
    assert "Never use raw_name or a weapon skin name as a teammate" in serialized


def test_summary_contract_rejects_switch_back_claims_that_conflict_with_equipment_timeline() -> None:
    manifest = _manifest(duration_sec=25.0)
    metadata = manifest.metadata
    metadata.user_metadata["qwen_visual_ocr"] = {
        "schema_version": "1.0",
        "source": "qwen35_visual_ocr",
        "equipment_timeline": [
            {
                "timestamp": 0.0,
                "entity_name": "Mosin Obrez (Rougarou skin)",
                "entity_type": "weapon",
                "source": "qwen35_visual_ocr",
                "confidence": 0.95,
            },
            {
                "timestamp": 19.0,
                "entity_name": "Auto-5",
                "entity_type": "weapon",
                "source": "qwen35_visual_ocr",
                "confidence": 0.98,
            },
        ],
    }
    raw = json.dumps(
        {
            "title": "Fight",
            "short_summary": "The player fires Auto-5 before switching back to the Mosin Obrez.",
            "detailed_summary": (
                "At 19.0 seconds, the player switches back to the Mosin Obrez (Rougarou skin). "
                "The player's weapon is back to the Mosin Obrez."
            ),
            "visual_observations": [
                {
                    "window_id": "window_full_clip",
                    "start": 19.0,
                    "end": 25.0,
                    "text": "The player switches back to the Mosin Obrez (Rougarou skin).",
                    "uncertainties": [],
                }
            ],
            "key_moments": [
                {
                    "start": 19.0,
                    "end": 25.0,
                    "description": "Player switches back to Mosin Obrez (Rougarou skin).",
                    "evidence": ["video"],
                    "evidence_pointers": [
                        {
                            "source": "video",
                            "window_id": "window_full_clip",
                            "start": 19.0,
                            "end": 25.0,
                            "quote_or_observation": "Player switches back to Mosin Obrez (Rougarou skin).",
                        }
                    ],
                }
            ],
            "tags": [],
            "detected_language": None,
            "uncertainties": [],
        }
    )
    with pytest.raises(ValueError, match="authoritative_equipment_timeline"):
        _parse_summary_contract(
            raw,
            manifest=manifest,
            model_id="Qwen/Qwen3.5-4B",
            metadata=metadata,
            weapon_resolver=lambda text: "Mosin Obrez (Rougarou skin)" if "rougarou" in text.lower() else None,
        )

    summary = parse_summary_json(raw, manifest=manifest, model_id="Qwen/Qwen3.5-4B")
    unchanged = _ensure_deterministic_observation_key_moments(
        summary,
        [],
        manifest,
        metadata=metadata,
        weapon_resolver=lambda text: "Mosin Obrez (Rougarou skin)" if "rougarou" in text.lower() else None,
    )
    public_text = json.dumps(unchanged.model_dump())

    assert "switches back" in public_text
    assert "weapon is back to" in public_text
    assert "Unsupported weapon-switch-back wording was repaired" not in unchanged.uncertainties


def test_summary_prompt_includes_weapon_skin_name_resolution_rules() -> None:
    manifest = _manifest(duration_sec=25.0)
    transcript = build_asr_transcript(
        manifest,
        model_id="openai/whisper-large-v3-turbo",
        text="",
        language=None,
        segments=[],
    )

    messages = _summary_messages(
        manifest,
        observations=[],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
        weapon_skin_map={"Rougarou": "Mosin Obrez (Rougarou skin)"},
    )
    serialized = json.dumps(messages)

    assert "weapon_name_resolution_rules" in serialized
    assert "Rougarou" in serialized
    assert "Mosin Obrez (Rougarou skin)" in serialized
    assert "do not write 'holding a Rougarou'" in serialized
    assert "output the corresponding display_name everywhere" in serialized


def test_combined_summary_prompt_does_not_duplicate_frame_paths_as_text() -> None:
    manifest = _manifest()
    transcript = build_asr_transcript(
        manifest,
        model_id="openai/whisper-large-v3-turbo",
        text="",
        language=None,
        segments=[],
    )
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=10.0,
        duration_sec=10.0,
        prepared_video_frame_paths=["/tmp/frame_0000.png"],
        prepared_video_sample_fps=2.0,
        prepared_video_metadata={
            "frame_paths": ["/tmp/frame_0000.png"],
            "qwen_video_frame_dir": "/tmp/qwen",
            "source_video_width": 1920,
        },
    )

    messages = _summary_messages(
        manifest,
        observations=[],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
        media_windows=[window],
        require_visual_observations=True,
        spec=model_for_role("summarizer"),
    )
    text_payload = messages[1]["content"][0]["text"]

    assert "visual_observations is required" in text_payload
    assert "at least one direct-video visual_observation" in text_payload
    assert "final engagement frames" in text_payload
    assert "16-21 second range" in text_payload
    assert "enemy position" in text_payload
    assert "source_video_width" in text_payload
    assert "frame_0000.png" not in text_payload
    assert "qwen_video_frame_dir" not in text_payload
    assert messages[1]["content"][1]["video"] == ["/tmp/frame_0000.png"]


def test_combined_summary_attaches_focused_engagement_crops(tmp_path: Path) -> None:
    from PIL import Image

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_paths = []
    timestamps = [15.95, 16.05, 16.75, 17.5, 18.083, 18.333, 20.0, 20.5]
    for index, timestamp in enumerate(timestamps):
        path = frame_dir / f"frame_{index:04d}.png"
        Image.new("RGB", (160, 90), (80 + index, 90, 100)).save(path)
        frame_paths.append(str(path))
    manifest = _manifest(duration_sec=25.0)
    manifest.metadata.user_metadata["hit_marker"] = {
        "detected": True,
        "timestamp": 18.083,
        "evidence": [{"timestamp": 18.083, "frame_path": frame_paths[4]}],
    }
    transcript = build_asr_transcript(
        manifest,
        model_id="openai/whisper-large-v3",
        text="",
        language=None,
        segments=[],
    )
    window = MediaWindowV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="window_full_clip",
        start_sec=0.0,
        end_sec=25.0,
        duration_sec=25.0,
        prepared_video_frame_paths=frame_paths,
        prepared_video_sample_fps=6.0,
        prepared_video_metadata={
            "qwen_video_frame_dir": str(frame_dir),
            "qwen_video_frame_timestamps_sec": timestamps,
        },
    )

    messages = _summary_messages(
        manifest,
        observations=[],
        transcript=transcript,
        audio_captions=[],
        metadata=manifest.metadata,
        media_windows=[window],
        require_visual_observations=True,
        spec=model_for_role("summarizer"),
    )

    content = messages[1]["content"]
    assert any(item.get("text", "").startswith("Focused engagement crops follow") for item in content if item["type"] == "text")
    focused_images = [item for item in content if item["type"] == "image" and "summary-focus-crops-v1" in item["image"]]
    assert focused_images
    assert focused_images[0]["max_pixels"] >= 262144


def test_summary_prompt_attaches_detected_death_screen_frame(tmp_path: Path) -> None:
    frame = tmp_path / "death_screen.jpg"
    frame.write_bytes(b"fake image")
    metadata = _metadata()
    metadata.user_metadata["death_screen"] = {
        "frame_path": str(frame),
        "timestamp": 7.25,
        "status": "killed",
        "killed_with": "Sparks Pistol",
        "confidence": 0.9,
        "raw_text": "YOU ARE DEAD",
    }
    manifest = ClipManifestV1(
        clip_id=metadata.clip_id,
        file_name=metadata.file_name,
        file_path=metadata.file_path,
        duration_sec=10.0,
        media_type="video",
        metadata=metadata,
        ingest_timestamp="2026-05-13T00:00:00Z",
    )
    transcript = build_asr_transcript(
        manifest,
        model_id="openai/whisper-large-v3-turbo",
        text="",
        language=None,
        segments=[],
    )

    messages = _summary_messages(
        manifest,
        observations=[],
        transcript=transcript,
        audio_captions=[],
        metadata=metadata,
    )
    content = messages[1]["content"]

    assert isinstance(content, list)
    assert content[1] == {"type": "image", "image": str(frame)}
    serialized = json.dumps(messages)
    assert "death_screen_visual_inputs" in serialized
    assert "death_screen_frame" in serialized
    assert "source label video" in serialized


def test_final_summary_schema_requires_evidence_pointers_and_bounds() -> None:
    pointer = EvidencePointerV1(
        source="video",
        window_id="window_001",
        start=0.0,
        end=2.0,
        quote_or_observation="Visible player movement near cover.",
    )
    summary = FusedSummaryV1(
        clip_id=7,
        file_name="round_01.mp4",
        model_id="Qwen/Qwen3.5-4B",
        title="round_01.mp4",
        short_summary="Visible player movement near cover.",
        detailed_summary="The clip shows visible player movement near cover.",
        key_moments=[
            KeyMomentV1(
                start=0.0,
                end=2.0,
                description="Player movement near cover.",
                evidence=["video"],
                evidence_pointers=[pointer],
            )
        ],
    )

    summary.validate_evidence_bounds(2.0)
    with pytest.raises(ValueError, match="clip duration"):
        summary.validate_evidence_bounds(1.0)


def test_audio_only_summary_claims_must_be_uncertain() -> None:
    summary = FusedSummaryV1(
        clip_id=7,
        file_name="round_01.mp4",
        model_id="Qwen/Qwen3.5-4B",
        title="round_01.mp4",
        short_summary="Footsteps may be audible.",
        detailed_summary="Footsteps may be audible.",
        key_moments=[
            KeyMomentV1(
                start=0.0,
                end=1.0,
                description="Footsteps may be audible.",
                evidence=["audio"],
                evidence_pointers=[
                    EvidencePointerV1(
                        source="audio",
                        window_id="window_001",
                        start=0.0,
                        end=1.0,
                        quote_or_observation="Possible footsteps are audible.",
                    )
                ],
            )
        ],
    )

    with pytest.raises(ValueError, match="audio-only"):
        summary.validate_evidence_bounds(2.0)


def test_parse_summary_json_rejects_out_of_bounds_evidence() -> None:
    raw = json.dumps(
        {
            "title": "round_01.mp4",
            "short_summary": "Movement.",
            "detailed_summary": "Movement.",
            "key_moments": [
                {
                    "start": 0.0,
                    "end": 11.0,
                    "description": "Movement.",
                    "evidence": ["video"],
                    "evidence_pointers": [
                        {
                            "source": "video",
                            "window_id": "window_001",
                            "start": 0.0,
                            "end": 11.0,
                            "quote_or_observation": "Movement.",
                        }
                    ],
                }
            ],
            "tags": ["movement"],
            "detected_language": "en",
            "uncertainties": [],
        }
    )

    with pytest.raises(ValueError, match="clip duration"):
        parse_summary_json(raw, manifest=_manifest(duration_sec=10.0), model_id="Qwen/Qwen3.5-4B")


def test_parse_summary_json_clamps_small_timestamp_boundary_drift() -> None:
    raw = json.dumps(
        {
            "title": "round_01.mp4",
            "short_summary": "Movement.",
            "detailed_summary": "Movement.",
            "key_moments": [
                {
                    "start": 0.0,
                    "end": 2.05,
                    "description": "Movement.",
                    "evidence": ["video"],
                    "evidence_pointers": [
                        {
                            "source": "video",
                            "window_id": "window_001",
                            "start": 0.0,
                            "end": 2.05,
                            "quote_or_observation": "Movement.",
                        }
                    ],
                }
            ],
            "tags": ["movement"],
            "detected_language": "en",
            "uncertainties": [],
        }
    )

    summary = parse_summary_json(raw, manifest=_manifest(duration_sec=2.008833), model_id="Qwen/Qwen3.5-4B")

    assert summary.key_moments[0].end == 2.008833
    assert summary.key_moments[0].evidence_pointers[0].end == 2.008833


def test_parse_summary_json_accepts_json_code_fence() -> None:
    raw = """```json
{
  "title": "round_01.mp4",
  "short_summary": "Movement.",
  "detailed_summary": "Movement supported by video evidence.",
  "key_moments": [
    {
      "start": 0.0,
      "end": 1.0,
      "description": "Movement.",
      "evidence": ["video"],
      "evidence_pointers": [
        {
          "source": "video",
          "window_id": "window_001",
          "start": 0.0,
          "end": 1.0,
          "quote_or_observation": "Movement."
        }
      ]
    }
  ],
  "tags": ["movement"],
  "detected_language": null,
  "uncertainties": []
}
```"""

    summary = parse_summary_json(raw, manifest=_manifest(duration_sec=10.0), model_id="Qwen/Qwen3.5-4B")

    assert summary.title == "round_01.mp4"


def test_parse_summary_json_uses_first_complete_json_object() -> None:
    raw = """prefix text
{
  "title": "round_01.mp4",
  "short_summary": "Movement.",
  "detailed_summary": "Movement supported by video evidence.",
  "key_moments": [
    {
      "start": 0.0,
      "end": 1.0,
      "description": "Movement.",
      "evidence": ["video"],
      "evidence_pointers": [
        {
          "source": "video",
          "window_id": "window_001",
          "start": 0.0,
          "end": 1.0,
          "quote_or_observation": "Movement."
        }
      ]
    }
  ],
  "tags": ["movement"],
  "detected_language": null,
  "uncertainties": []
}
trailing repeated text {"title": "ignored"}"""

    summary = parse_summary_json(raw, manifest=_manifest(duration_sec=10.0), model_id="Qwen/Qwen3.5-4B")

    assert summary.title == "round_01.mp4"


def test_parse_summary_json_drops_invalid_evidence_labels_when_pointer_is_valid() -> None:
    raw = json.dumps(
        {
            "title": "round_01.mp4",
            "short_summary": "Movement.",
            "detailed_summary": "Movement supported by video evidence.",
            "key_moments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "description": "Movement.",
                    "evidence": ["video", "focus"],
                    "evidence_pointers": [
                        {
                            "source": "video",
                            "window_id": "window_001",
                            "start": 0.0,
                            "end": 1.0,
                            "quote_or_observation": "Movement.",
                        }
                    ],
                }
            ],
            "tags": ["movement"],
            "detected_language": None,
            "uncertainties": [],
        }
    )

    summary = parse_summary_json(raw, manifest=_manifest(duration_sec=10.0), model_id="Qwen/Qwen3.5-4B")

    assert summary.key_moments[0].evidence == ["video"]
    assert summary.key_moments[0].evidence_pointers[0].source == "video"


def test_parse_summary_json_coerces_pointer_objects_from_evidence_field() -> None:
    raw = json.dumps(
        {
            "title": "round_01.mp4",
            "short_summary": "Movement.",
            "detailed_summary": "Movement supported by video evidence.",
            "key_moments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "description": "Movement.",
                    "evidence": [
                        {
                            "source": "video",
                            "window_id": "window_001",
                            "start": 0.0,
                            "end": 1.0,
                            "quote_or_observation": "Movement.",
                        }
                    ],
                }
            ],
            "tags": ["movement"],
            "detected_language": None,
            "uncertainties": [],
        }
    )

    summary = parse_summary_json(raw, manifest=_manifest(duration_sec=10.0), model_id="Qwen/Qwen3.5-4B")

    assert summary.key_moments[0].evidence == ["video"]
    assert summary.key_moments[0].evidence_pointers[0].quote_or_observation == "Movement."
