from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .audio import extract_audio, media_cache_key
from .qwen_video import _copy_frame_image, _is_hdr_or_10bit, _prepare_and_save_frame, probe_video_pyav
from ..config import IndexingSettings
from ..db import Database
from ..runtime_tools import resolve_ffmpeg


@dataclass(frozen=True)
class SegmentSpec:
    source_path: Path
    start_seconds: float
    duration_seconds: float
    label: str | None = None
    segment_id: str | None = None


@dataclass(frozen=True)
class SegmentPaths:
    segment_dir: Path
    video_path: Path
    audio_path: Path
    metadata_path: Path
    frames_dir: Path


@dataclass(frozen=True)
class SegmentExtractionResult:
    spec: SegmentSpec
    paths: SegmentPaths
    success: bool
    reused: bool = False
    errors: tuple[str, ...] = ()
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameExtractionResult:
    video_path: Path
    frames_dir: Path
    frame_paths: tuple[Path, ...]
    success: bool
    error: str | None = None
    source: str = "opencv"


@dataclass(frozen=True)
class ClipSegmentExtractionSummary:
    clip_id: int
    total_segments: int
    completed_segments: int
    failed_segments: int
    skipped_segments: int = 0


def build_segment_id(
    source_path: str | Path,
    start_seconds: float,
    duration_seconds: float,
    label: str | None = None,
) -> str:
    payload = f"{Path(source_path).expanduser().resolve(strict=False)}:{start_seconds:.3f}:{duration_seconds:.3f}:{label or ''}"
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def segment_cache_layout(
    cache_root: str | Path,
    source_path: str | Path,
    start_seconds: float,
    duration_seconds: float,
    *,
    label: str | None = None,
    segment_id: str | None = None,
) -> SegmentPaths:
    segment_key = segment_id or build_segment_id(source_path, start_seconds, duration_seconds, label)
    segment_dir = Path(cache_root) / "segments" / media_cache_key(source_path) / segment_key
    return SegmentPaths(
        segment_dir=segment_dir,
        video_path=segment_dir / "clip.mp4",
        audio_path=segment_dir / "audio.wav",
        metadata_path=segment_dir / "metadata.json",
        frames_dir=segment_dir / "frames",
    )


def write_segment_metadata(
    paths: SegmentPaths,
    spec: SegmentSpec,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    paths.segment_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "source_path": str(spec.source_path),
        "start_seconds": spec.start_seconds,
        "duration_seconds": spec.duration_seconds,
        "label": spec.label,
        "segment_id": spec.segment_id,
        "video_path": str(paths.video_path),
        "audio_path": str(paths.audio_path),
        "frames_dir": str(paths.frames_dir),
    }
    if extra:
        payload.update(extra)
    paths.metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def extract_av_segment(
    spec: SegmentSpec,
    cache_root: str | Path,
    *,
    overwrite: bool = False,
    extract_audio_track: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    timeout: float = 60.0,
) -> SegmentExtractionResult:
    if spec.start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    if spec.duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    paths = segment_cache_layout(
        cache_root,
        spec.source_path,
        spec.start_seconds,
        spec.duration_seconds,
        label=spec.label,
        segment_id=spec.segment_id,
    )

    if paths.video_path.exists() and paths.metadata_path.exists() and not overwrite:
        return SegmentExtractionResult(spec=spec, paths=paths, success=True, reused=True)

    resolved_ffmpeg = resolve_ffmpeg(ffmpeg_bin)
    if resolved_ffmpeg is None:
        write_segment_metadata(paths, spec, extra={"success": False, "errors": [f"{ffmpeg_bin} not found"]})
        return SegmentExtractionResult(spec=spec, paths=paths, success=False, errors=(f"{ffmpeg_bin} not found",))

    paths.segment_dir.mkdir(parents=True, exist_ok=True)
    command = [
        resolved_ffmpeg,
        "-y" if overwrite else "-n",
        "-ss",
        f"{spec.start_seconds:.3f}",
        "-i",
        str(spec.source_path),
        "-t",
        f"{spec.duration_seconds:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        str(paths.video_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors = (str(exc),)
        write_segment_metadata(paths, spec, extra={"success": False, "errors": list(errors)})
        return SegmentExtractionResult(spec=spec, paths=paths, success=False, errors=errors, command=tuple(command))

    errors: list[str] = []
    if completed.returncode != 0:
        errors.append((completed.stderr or completed.stdout).strip() or "ffmpeg failed")

    if extract_audio_track and not errors:
        audio_result = extract_audio(
            paths.video_path,
            paths.segment_dir,
            sample_rate=16_000,
            channels=1,
            overwrite=overwrite,
            ffmpeg_bin=resolved_ffmpeg,
            timeout=timeout,
        )
        if audio_result.output_path != paths.audio_path and audio_result.output_path.exists():
            paths.audio_path.parent.mkdir(parents=True, exist_ok=True)
            if paths.audio_path.exists() and overwrite:
                paths.audio_path.unlink()
            if not paths.audio_path.exists():
                audio_result.output_path.replace(paths.audio_path)
        if not audio_result.success:
            errors.append(audio_result.error or "audio extraction failed")

    success = not errors
    write_segment_metadata(paths, spec, extra={"success": success, "errors": errors, "command": command})
    return SegmentExtractionResult(
        spec=spec,
        paths=paths,
        success=success,
        errors=tuple(errors),
        command=tuple(command),
    )


def extract_frames_opencv(
    video_path: str | Path,
    frames_dir: str | Path,
    *,
    every_n_frames: int = 30,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> FrameExtractionResult:
    if every_n_frames <= 0:
        raise ValueError("every_n_frames must be positive")

    try:
        import cv2  # type: ignore
    except ImportError:
        return FrameExtractionResult(
            video_path=Path(video_path),
            frames_dir=Path(frames_dir),
            frame_paths=(),
            success=False,
            error="opencv not installed",
        )

    output_dir = Path(frames_dir)
    existing = tuple(sorted(output_dir.glob("frame_*.jpg")))
    if existing and not overwrite:
        return FrameExtractionResult(Path(video_path), output_dir, existing, success=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for frame_path in output_dir.glob("frame_*.jpg"):
            frame_path.unlink()

    capture = cv2.VideoCapture(str(video_path))
    frame_paths: list[Path] = []
    try:
        if not capture.isOpened():
            return FrameExtractionResult(Path(video_path), output_dir, (), success=False, error="could not open video")

        index = 0
        saved = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every_n_frames == 0:
                frame_path = output_dir / f"frame_{saved:06d}.jpg"
                if cv2.imwrite(str(frame_path), frame):
                    frame_paths.append(frame_path)
                    saved += 1
                if max_frames is not None and saved >= max_frames:
                    break
            index += 1
    finally:
        capture.release()

    return FrameExtractionResult(Path(video_path), output_dir, tuple(frame_paths), success=bool(frame_paths))


def build_segment_specs(
    source_path: str | Path,
    duration_seconds: float | None,
    *,
    segment_seconds: float,
    stride_seconds: float,
    settings_hash: str,
) -> list[SegmentSpec]:
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    if duration_seconds is None or duration_seconds <= 0:
        duration_seconds = segment_seconds

    specs: list[SegmentSpec] = []
    start = 0.0
    index = 0
    while start < duration_seconds:
        end = min(duration_seconds, start + segment_seconds)
        actual_duration = max(0.05, end - start)
        segment_id = build_segment_id(source_path, start, actual_duration, settings_hash)
        specs.append(
            SegmentSpec(
                source_path=Path(source_path),
                start_seconds=round(start, 3),
                duration_seconds=round(actual_duration, 3),
                label=f"{settings_hash}-{index:05d}",
                segment_id=segment_id,
            )
        )
        index += 1
        if end >= duration_seconds:
            break
        start += stride_seconds
    return specs


def extract_clip_segments(
    db: Database,
    clip_row: Any,
    indexing: IndexingSettings,
    cache_root: str | Path,
    *,
    overwrite: bool = False,
    progress_callback: object | None = None,
) -> ClipSegmentExtractionSummary:
    settings_hash = indexing.segment_settings_hash()
    duration = _row_get(clip_row, "duration")
    clip_id = int(_row_get(clip_row, "id"))
    source_path = Path(str(_row_get(clip_row, "path")))
    group_name = str(_row_get(clip_row, "group_name") or "Ungrouped")
    specs = build_segment_specs(
        source_path,
        float(duration) if duration is not None else None,
        segment_seconds=indexing.segment_seconds,
        stride_seconds=indexing.segment_stride_seconds,
        settings_hash=settings_hash,
    )

    completed = 0
    failed = 0
    skipped = 0
    for index, spec in enumerate(specs, start=1):
        if progress_callback and callable(progress_callback):
            progress_callback(
                message=f"Extracting segment {index}/{len(specs)}",
                progress=index / max(len(specs), 1),
                data={"clip_id": clip_id, "stage": "audio-video segment extraction"},
            )

        paths = segment_cache_layout(cache_root, source_path, spec.start_seconds, spec.duration_seconds, label=spec.label, segment_id=spec.segment_id)
        paths.segment_dir.mkdir(parents=True, exist_ok=True)
        paths.frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = extract_representative_frames(
            source_path,
            paths.frames_dir,
            start_seconds=spec.start_seconds,
            duration_seconds=spec.duration_seconds,
            frame_count=indexing.representative_frames_per_segment,
            overwrite=overwrite,
        )
        audio_path: Path | None = None
        video_path: Path | None = None
        errors: list[str] = []

        if indexing.store_audio_segment_files:
            audio_ok, audio_error = extract_audio_segment_to_path(
                source_path,
                paths.audio_path,
                start_seconds=spec.start_seconds,
                duration_seconds=spec.duration_seconds,
                overwrite=overwrite,
            )
            if audio_ok:
                audio_path = paths.audio_path
            elif audio_error:
                errors.append(audio_error)

        if indexing.store_video_segment_files:
            result = extract_av_segment(spec, cache_root, overwrite=overwrite, extract_audio_track=False)
            if result.success:
                video_path = result.paths.video_path
            else:
                errors.extend(result.errors)

        modality = "audio_video" if frame_paths and audio_path else "video_only" if frame_paths else "audio_only" if audio_path else "video_only"
        segment_id = db.upsert_segment(
            {
                "clip_id": clip_id,
                "group_name": group_name,
                "start_time": spec.start_seconds,
                "end_time": round(spec.start_seconds + spec.duration_seconds, 3),
                "duration": spec.duration_seconds,
                "modality": modality,
                "representative_frame_path": str(frame_paths[0]) if frame_paths else None,
                "video_segment_path": str(video_path) if video_path else None,
                "audio_segment_path": str(audio_path) if audio_path else None,
                "segment_settings_hash": settings_hash,
                "error_message": "; ".join(errors) if errors and not (frame_paths or audio_path) else None,
            }
        )
        for frame_index, frame_path in enumerate(frame_paths, start=1):
            timestamp = spec.start_seconds
            if len(frame_paths) > 1:
                timestamp += (spec.duration_seconds * (frame_index - 1)) / (len(frame_paths) - 1)
            db.add_segment_frame(segment_id, str(frame_path), round(timestamp, 3), frame_index)
        write_segment_metadata(
            paths,
            spec,
            extra={
                "clip_id": clip_id,
                "group_name": group_name,
                "modality": modality,
                "frames": [str(path) for path in frame_paths],
                "audio_path": str(audio_path) if audio_path else None,
                "video_path": str(video_path) if video_path else None,
                "errors": errors,
            },
        )
        if frame_paths or audio_path or video_path:
            completed += 1
        else:
            failed += 1

    return ClipSegmentExtractionSummary(clip_id=clip_id, total_segments=len(specs), completed_segments=completed, failed_segments=failed, skipped_segments=skipped)


def extract_representative_frames(
    video_path: str | Path,
    frames_dir: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    frame_count: int,
    overwrite: bool = False,
    ffmpeg_bin: str = "ffmpeg",
) -> list[Path]:
    del ffmpeg_bin
    frame_count = max(1, frame_count)
    output_dir = Path(frames_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_representative_frames(output_dir)
    if existing and not overwrite:
        return existing[:frame_count]
    if overwrite:
        for existing_frame in _existing_representative_frames(output_dir):
            existing_frame.unlink()

    try:
        import av
    except ModuleNotFoundError:
        return []

    source = Path(video_path).expanduser().resolve()
    try:
        probe = probe_video_pyav(source)
    except Exception:
        return []

    duration = max(0.0, float(duration_seconds or 0.0))
    start = max(0.0, float(start_seconds or 0.0))
    offsets = [duration / 2.0] if frame_count == 1 else [
        (duration * index) / max(frame_count - 1, 1) for index in range(frame_count)
    ]
    target_times = [start + offset for offset in offsets]
    frame_paths: list[Path] = []

    try:
        container = av.open(str(source))
    except Exception:
        return []

    hdr_or_10bit = _is_hdr_or_10bit(probe)
    next_index = 0
    with container:
        stream = container.streams.video[0]
        try:
            stream.thread_type = "AUTO"
        except Exception:
            pass
        if target_times:
            try:
                container.seek(int(max(0.0, target_times[0] - 0.1) * av.time_base), any_frame=False, backward=True)
            except Exception:
                pass
        for frame in container.decode(stream):
            if next_index >= len(target_times):
                break
            timestamp = _pyav_frame_time(frame)
            if timestamp + 1e-6 < target_times[next_index]:
                continue
            image = frame.to_image().convert("RGB")
            try:
                while next_index < len(target_times) and timestamp + 1e-6 >= target_times[next_index]:
                    frame_path = output_dir / f"frame_{next_index + 1:03d}.png"
                    _prepare_and_save_frame(
                        image.copy(),
                        frame_path,
                        probe=probe,
                        hdr_or_10bit=hdr_or_10bit,
                        max_width=1280,
                    )
                    frame_paths.append(frame_path)
                    next_index += 1
            finally:
                image.close()
    while frame_paths and len(frame_paths) < frame_count:
        frame_path = output_dir / f"frame_{len(frame_paths) + 1:03d}.png"
        _copy_frame_image(frame_paths[-1], frame_path)
        frame_paths.append(frame_path)
    return frame_paths


def _existing_representative_frames(output_dir: Path) -> list[Path]:
    return sorted([*output_dir.glob("frame_*.png"), *output_dir.glob("frame_*.jpg")])


def _pyav_frame_time(frame: Any) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and frame.time_base is not None:
        return float(frame.pts * frame.time_base)
    return 0.0


def extract_audio_segment_to_path(
    media_path: str | Path,
    output_path: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    overwrite: bool = False,
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[bool, str | None]:
    output = Path(output_path)
    if output.exists() and not overwrite:
        return True, None
    resolved_ffmpeg = resolve_ffmpeg(ffmpeg_bin)
    if resolved_ffmpeg is None:
        return False, f"{ffmpeg_bin} not found"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        resolved_ffmpeg,
        "-y" if overwrite else "-n",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(media_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout).strip() or "ffmpeg audio extraction failed"
    return output.exists(), None if output.exists() else "audio output was not created"


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except Exception:
        return getattr(row, key)
