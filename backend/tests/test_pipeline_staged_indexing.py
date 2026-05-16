from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from backend.app import pipeline
from backend.app.config import AppSettings
from backend.app.db import Database
from backend.app.hf_pipeline.schemas import (
    AudioCaptionV1,
    EvidencePointerV1,
    FusedSummaryV1,
    KeyMomentV1,
    VideoObservationV1,
)
from backend.app.models import ScanSummary
from backend.app.processing.qwen_video import QwenVideoInput


def test_run_indexing_processes_all_clips_by_model_stage(tmp_path, monkeypatch) -> None:
    settings = AppSettings(
        clips_dir=tmp_path / "clips",
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        gpu_backend="cpu",
        allow_mock_models=True,
        auto_download_models=False,
        enable_transcription=True,
        qdrant_url="local",
    )
    settings.ensure_dirs()
    events: list[tuple[str, int | str | float]] = []

    def fake_scan_directory(root, db, **kwargs):  # noqa: ANN001, ANN003, ANN202
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for clip_number in (1, 2):
            db.upsert_clip(
                {
                    "file_hash": f"hash-{clip_number}",
                    "filename": f"clip_{clip_number}.mp4",
                    "path": str(root / f"clip_{clip_number}.mp4"),
                    "relative_path": f"clip_{clip_number}.mp4",
                    "source_root": str(root),
                    "group_name": "Test",
                    "duration": 8.0,
                    "size_bytes": 1024,
                    "width": 1920,
                    "height": 1080,
                    "fps": 60.0,
                    "codec": "h264",
                    "status": "pending",
                    "scan_status": "new",
                    "summary": "",
                }
            )
        return ScanSummary(source_root=str(root), files_seen=2, files_new=2, supported_videos=2, groups={"Test": 2})

    def fake_extract_clip_segments(db, clip, indexing, data_dir, **kwargs):  # noqa: ANN001, ANN003, ANN202
        clip_id = int(clip["id"])
        events.append(("prepare", clip_id))
        db.upsert_segment(
            {
                "clip_id": clip_id,
                "group_name": "Test",
                "start_time": 0.0,
                "end_time": 8.0,
                "duration": 8.0,
                "modality": "video_audio",
                "representative_frame_path": None,
                "video_segment_path": str(Path(data_dir) / f"clip_{clip_id}_window.mp4"),
                "audio_segment_path": str(Path(data_dir) / f"clip_{clip_id}.wav"),
                "segment_settings_hash": indexing.segment_settings_hash(),
            }
        )
        return SimpleNamespace(total_segments=1)

    def fake_extract_audio_segment_to_path(media_path, output_path, **kwargs):  # noqa: ANN001, ANN003, ANN202
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).touch()
        events.append(("audio_window", float(kwargs["start_seconds"])))
        return True, None

    class FakeManager:
        def unload(self) -> None:
            events.append(("unload", "manager"))

    class FakeTranscriber:
        def transcribe(self, audio_source):  # noqa: ANN001, ANN202
            clip_id = int(re.search(r"clip_(\d+)", str(audio_source)).group(1))
            events.append(("asr", clip_id))
            return SimpleNamespace(
                text=f"speech for clip {clip_id}",
                engine=settings.tier.asr_model,
                language="en",
                segments=[
                    SimpleNamespace(
                        start=0.0,
                        end=1.0,
                        text=f"speech for clip {clip_id}",
                        confidence=0.9,
                    )
                ],
            )

    class FakeAudioCaptioner:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.window_sec = 5.0
            self.stride_sec = 2.5

        def caption(self, manifest, window):  # noqa: ANN001, ANN202
            events.append(("audio", int(manifest.clip_id)))
            return AudioCaptionV1(
                clip_id=manifest.clip_id,
                file_name=manifest.file_name,
                window_id=window.window_id,
                start_sec=window.start_sec,
                end_sec=window.end_sec,
                model_id=settings.tier.audio_captioner_model,
                text=f"possible non-speech audio for clip {manifest.clip_id}",
            )

        def caption_windows(self, manifest, windows):  # noqa: ANN001, ANN202
            return [self.caption(manifest, window) for window in windows]

    class FakeFusionSummarizer:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def extract_visual_ocr(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            return {}

        def observe_video_windows(self, manifest, windows):  # noqa: ANN001, ANN202
            events.append(("observe", int(manifest.clip_id)))
            return [
                VideoObservationV1(
                    clip_id=manifest.clip_id,
                    file_name=manifest.file_name,
                    window_id=windows[0].window_id,
                    start_sec=0.0,
                    end_sec=2.0,
                    model_id=settings.tier.deep_reasoning_model,
                    text=f"video observation for clip {manifest.clip_id}",
                )
            ]

        def summarize(self, manifest, **kwargs):  # noqa: ANN001, ANN003, ANN202
            clip_id = int(manifest.clip_id)
            events.append(("summarize", clip_id))
            pointer = EvidencePointerV1(
                source="video",
                window_id="window_000001",
                start=0.0,
                end=1.0,
                quote_or_observation=f"video observation for clip {clip_id}",
            )
            return FusedSummaryV1(
                clip_id=clip_id,
                file_name=manifest.file_name,
                model_id=settings.tier.deep_reasoning_model,
                title=manifest.file_name,
                short_summary=f"summary for clip {clip_id}",
                detailed_summary=f"detailed summary for clip {clip_id}",
                key_moments=[
                    KeyMomentV1(
                        start=0.0,
                        end=1.0,
                        description=f"moment for clip {clip_id}",
                        evidence=["video"],
                        evidence_pointers=[pointer],
                    )
                ],
                tags=[f"clip-{clip_id}"],
            )

    class FakeEmbedder:
        dimension = 8
        uses_real_backend = False
        config = SimpleNamespace(model_name=settings.tier.multimodal_retrieval_model)

        def embed_video_frames(self, frame_paths, **kwargs):  # noqa: ANN001, ANN003, ANN202
            clip_id = int(Path(frame_paths[0]).stem.split("_")[1].split(".")[0])
            events.append(("embed_video", clip_id))
            return [1.0] * self.dimension

        def embed_video_path(self, video_path, **kwargs):  # noqa: ANN001, ANN003, ANN202
            clip_id = int(Path(video_path).stem.split("_")[1])
            events.append(("embed_video", clip_id))
            return [1.0] * self.dimension

        def embed_text(self, text):  # noqa: ANN001, ANN202
            events.append(("embed_text", "text"))
            return [0.5] * self.dimension

        def embed_query(self, text):  # noqa: ANN001, ANN202
            return [0.5] * self.dimension

    class FakeVectorStore:
        using_qdrant = False

        def __init__(self) -> None:
            self.records: list[tuple[int, str]] = []

        def delete_by_clip_id(self, clip_id):  # noqa: ANN001
            events.append(("delete_vectors", int(clip_id)))

        def add_vector(self, field, vector_id, vector, payload):  # noqa: ANN001
            self.records.append((int(payload["clip_id"]), str(field)))

    fake_store = FakeVectorStore()
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "scan_directory", fake_scan_directory)
    monkeypatch.setattr(pipeline, "extract_clip_segments", fake_extract_clip_segments)
    monkeypatch.setattr(pipeline, "extract_audio_segment_to_path", fake_extract_audio_segment_to_path)
    monkeypatch.setattr(pipeline, "_model_runtime_manager", lambda _settings: FakeManager())
    monkeypatch.setattr(pipeline, "_transcriber", lambda _settings: FakeTranscriber())
    monkeypatch.setattr(pipeline, "AudioCaptionerAdapter", FakeAudioCaptioner)
    monkeypatch.setattr(pipeline, "FusionSummarizerAdapter", FakeFusionSummarizer)
    monkeypatch.setattr(pipeline, "_embedder", lambda _settings: FakeEmbedder())
    monkeypatch.setattr(pipeline, "_vector_store", lambda _settings, _dimension: fake_store)
    monkeypatch.setattr(
        pipeline,
        "prepare_qwen_video_input",
        lambda video_path, *args, **kwargs: QwenVideoInput(
            source_path=str(video_path),
            frame_paths=[str(Path(video_path).with_suffix(".frame_0000.png"))],
            mode="sampled_sdr_frame_sequence",
            sample_fps=1.0,
            metadata={"qwen_video_input_mode": "sampled_sdr_frame_sequence"},
        ),
    )

    result = pipeline.run_indexing(source=str(settings.clips_dir), force=True)

    assert result["completed"] == 2
    assert result["failed"] == 0
    assert [item[1] for item in events if item[0] == "prepare"] == [1, 2]
    assert [item[1] for item in events if item[0] == "asr"] == [1, 2]
    assert [item[1] for item in events if item[0] == "audio"] == [1, 2]
    assert [item[1] for item in events if item[0] == "observe"] == [1, 2]
    assert [item[1] for item in events if item[0] == "summarize"] == [1, 2]
    assert [item[1] for item in events if item[0] == "embed_video"] == [1, 2]
    _assert_stage_before(events, "prepare", "asr")
    _assert_stage_before(events, "asr", "audio")
    _assert_stage_before(events, "audio", "observe")
    _assert_stage_before(events, "summarize", "embed_video")
    assert {
        field
        for clip_id, field in fake_store.records
        if clip_id == 1
    } == {"video", "summary", "speech", "audio_caption", "metadata", "fused"}


def test_audio_caption_window_ranges_overlap_and_cover_tail() -> None:
    ranges = pipeline._audio_caption_window_ranges(25.066, window_sec=5.0, stride_sec=2.5)

    assert ranges[0] == (0.0, 5.0)
    assert ranges[1] == (2.5, 7.5)
    assert ranges[-1][1] == 25.066
    assert all(0.0 <= start < end for start, end in ranges)
    assert all(end - start <= 5.001 for start, end in ranges)
    assert len(ranges) > 1


def _assert_stage_before(events: list[tuple[str, int | str | float]], earlier: str, later: str) -> None:
    earlier_positions = [index for index, item in enumerate(events) if item[0] == earlier]
    later_positions = [index for index, item in enumerate(events) if item[0] == later]
    assert earlier_positions
    assert later_positions
    assert max(earlier_positions) < min(later_positions)
