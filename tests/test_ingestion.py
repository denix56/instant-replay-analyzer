from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.ingestion.scanner import detect_clip_groups, discover_files, scan_library
from backend.app.ingestion.video_metadata import metadata_from_ffprobe, read_video_metadata


def test_discover_files_is_deterministic_and_hashes_duplicates_only(tmp_path: Path) -> None:
    first = tmp_path / "a" / "round.mp4"
    second = tmp_path / "b" / "round.mp4"
    sidecar = tmp_path / "a" / "round.json"
    hidden = tmp_path / ".cache" / "hidden.mp4"

    first.parent.mkdir()
    second.parent.mkdir()
    hidden.parent.mkdir()
    first.write_bytes(b"same-bytes")
    second.write_bytes(b"same-bytes")
    sidecar.write_text("{}", encoding="utf-8")
    hidden.write_bytes(b"hidden")

    first_scan = discover_files(tmp_path, hash_policy="duplicates")
    second_scan = discover_files(tmp_path, hash_policy="duplicates")

    assert [item.path for item in first_scan] == [item.path for item in second_scan]
    assert hidden not in [item.path for item in first_scan]
    hashes = {item.path: item.sha256 for item in first_scan}
    assert hashes[first] is not None
    assert hashes[second] == hashes[first]
    assert hashes[sidecar] is None


def test_scan_library_groups_video_with_sidecars(tmp_path: Path) -> None:
    video = tmp_path / "kills_001.mp4"
    subtitle = tmp_path / "kills_001.vtt"
    notes = tmp_path / "notes.txt"
    video.write_bytes(b"video")
    subtitle.write_text("WEBVTT", encoding="utf-8")
    notes.write_text("not a clip", encoding="utf-8")

    result = scan_library(tmp_path, hash_policy="never")
    groups = detect_clip_groups(result.files)

    assert result.groups == groups
    assert len(result.groups) == 1
    assert result.groups[0].primary_video == video.resolve()
    assert result.groups[0].related_files == (subtitle.resolve(),)


def test_video_metadata_parses_ffprobe_payload() -> None:
    payload = {
        "format": {"duration": "12.500"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "60000/1001",
                "nb_frames": "750",
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }

    metadata = metadata_from_ffprobe("clip.mp4", payload)

    assert metadata.source == "ffprobe"
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert round(metadata.fps or 0, 3) == 59.94
    assert metadata.frame_count == 750
    assert metadata.duration_seconds == 12.5
    assert metadata.has_audio is True


def test_video_metadata_file_fallback_for_unprobeable_file(tmp_path: Path) -> None:
    video = tmp_path / "empty.mp4"
    video.write_bytes(b"")

    metadata = read_video_metadata(video, ffprobe_bin="definitely_missing_ffprobe")

    assert metadata.path == video
    assert metadata.source in {"file", "opencv"}
    if metadata.source == "file":
        assert metadata.raw["size_bytes"] == 0
