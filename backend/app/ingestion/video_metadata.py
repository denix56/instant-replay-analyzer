from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    has_audio: bool = False
    source: str = "file"
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def probe_ffprobe(
    path: str | Path,
    *,
    ffprobe_bin: str = "ffprobe",
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    if shutil.which(ffprobe_bin) is None:
        return None

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if completed.returncode != 0:
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def metadata_from_ffprobe(path: str | Path, payload: dict[str, Any]) -> VideoMetadata:
    streams = payload.get("streams") or []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    duration = _float_or_none(video_stream.get("duration"))
    if duration is None:
        duration = _float_or_none((payload.get("format") or {}).get("duration"))

    return VideoMetadata(
        path=Path(path),
        duration_seconds=duration,
        width=_int_or_none(video_stream.get("width")),
        height=_int_or_none(video_stream.get("height")),
        fps=_parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
        frame_count=_int_or_none(video_stream.get("nb_frames")),
        video_codec=video_stream.get("codec_name"),
        audio_codec=audio_stream.get("codec_name"),
        has_audio=bool(audio_stream),
        source="ffprobe",
        raw=payload,
    )


def metadata_from_opencv(path: str | Path) -> VideoMetadata | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
        duration = None
        if fps and frame_count:
            duration = frame_count / fps
        return VideoMetadata(
            path=Path(path),
            duration_seconds=duration,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None,
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None,
            fps=fps,
            frame_count=frame_count,
            source="opencv",
        )
    finally:
        capture.release()


def read_video_metadata(
    path: str | Path,
    *,
    prefer_ffprobe: bool = True,
    ffprobe_bin: str = "ffprobe",
) -> VideoMetadata:
    video_path = Path(path)
    if prefer_ffprobe:
        payload = probe_ffprobe(video_path, ffprobe_bin=ffprobe_bin)
        if payload is not None:
            return metadata_from_ffprobe(video_path, payload)

    opencv_metadata = metadata_from_opencv(video_path)
    if opencv_metadata is not None:
        return opencv_metadata

    try:
        stat = video_path.stat()
    except OSError as exc:
        return VideoMetadata(path=video_path, source="unavailable", error=str(exc))

    return VideoMetadata(
        path=video_path,
        source="file",
        raw={"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
    )


def _parse_fps(value: Any) -> float | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        return float(value) or None
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_float = _float_or_none(denominator)
        if not denominator_float:
            return None
        numerator_float = _float_or_none(numerator)
        return None if numerator_float is None else numerator_float / denominator_float
    return _float_or_none(text)


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
