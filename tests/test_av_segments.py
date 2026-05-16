from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.processing.av_segments import (
    SegmentSpec,
    build_segment_id,
    extract_av_segment,
    extract_frames_opencv,
    extract_representative_frames,
    segment_cache_layout,
)


def test_segment_cache_layout_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "match.mp4"
    source.write_bytes(b"fake")

    first = segment_cache_layout(tmp_path / "cache", source, 1.25, 3.5, label="goal")
    second = segment_cache_layout(tmp_path / "cache", source, 1.25, 3.5, label="goal")

    assert first == second
    assert first.video_path.name == "clip.mp4"
    assert first.audio_path.name == "audio.wav"
    assert first.metadata_path.name == "metadata.json"
    assert first.frames_dir.name == "frames"


def test_build_segment_id_changes_with_timing(tmp_path: Path) -> None:
    source = tmp_path / "match.mp4"

    assert build_segment_id(source, 0.0, 5.0) != build_segment_id(source, 1.0, 5.0)


def test_extract_av_segment_writes_failure_metadata_when_ffmpeg_missing(tmp_path: Path) -> None:
    source = tmp_path / "match.mp4"
    source.write_bytes(b"fake")
    spec = SegmentSpec(source_path=source, start_seconds=2.0, duration_seconds=4.0, label="fight")

    result = extract_av_segment(spec, tmp_path / "cache", ffmpeg_bin="definitely_missing_ffmpeg")

    assert result.success is False
    assert result.errors == ("definitely_missing_ffmpeg not found",)
    assert result.paths.metadata_path.exists()
    metadata = json.loads(result.paths.metadata_path.read_text(encoding="utf-8"))
    assert metadata["success"] is False
    assert metadata["start_seconds"] == 2.0
    assert metadata["duration_seconds"] == 4.0


def test_extract_frames_opencv_fails_gracefully_for_unreadable_file(tmp_path: Path) -> None:
    video = tmp_path / "not-a-video.mp4"
    video.write_bytes(b"fake")

    result = extract_frames_opencv(video, tmp_path / "frames", max_frames=1)

    assert result.success is False
    assert result.frame_paths == ()
    assert result.error in {"opencv not installed", "could not open video"}


def test_extract_representative_frames_uses_pyav_without_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("av")
    source = tmp_path / "source.mp4"
    _write_test_video(source)
    monkeypatch.setattr(
        "backend.app.processing.av_segments.resolve_ffmpeg",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ffmpeg should not be used")),
    )

    frames = extract_representative_frames(
        source,
        tmp_path / "frames",
        start_seconds=0.0,
        duration_seconds=2.0,
        frame_count=3,
        ffmpeg_bin="definitely_missing_ffmpeg",
    )

    assert len(frames) == 3
    assert all(path.suffix == ".png" for path in frames)
    for path in frames:
        with Image.open(path) as image:
            assert image.mode == "RGB"
            assert image.width <= 1280


def _write_test_video(path: Path) -> None:
    import av

    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=8)
    stream.width = 96
    stream.height = 54
    stream.pix_fmt = "yuv420p"
    for index in range(16):
        frame_rgb = np.zeros((54, 96, 3), dtype=np.uint8)
        frame_rgb[:, :, 0] = min(index * 14, 255)
        frame_rgb[:, :, 1] = np.linspace(0, 255, 96, dtype=np.uint8)
        frame_rgb[:, :, 2] = 128
        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
