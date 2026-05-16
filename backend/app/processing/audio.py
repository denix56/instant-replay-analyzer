from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from ..runtime_tools import resolve_ffmpeg


@dataclass(frozen=True)
class WavInfo:
    path: Path
    sample_rate: int
    channels: int
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class AudioExtractionResult:
    input_path: Path
    output_path: Path
    success: bool
    reused: bool = False
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    command: tuple[str, ...] = ()
    error: str | None = None
    source: str = "ffmpeg"


def media_cache_key(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=False)
    return sha256(str(resolved).encode("utf-8")).hexdigest()[:16]


def audio_cache_path(
    media_path: str | Path,
    cache_root: str | Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    extension: str = ".wav",
) -> Path:
    suffix = extension if extension.startswith(".") else f".{extension}"
    channel_name = "mono" if channels == 1 else f"{channels}ch"
    source = Path(media_path)
    filename = f"{source.stem}__sr{sample_rate}_{channel_name}{suffix}"
    return Path(cache_root) / "audio" / media_cache_key(media_path) / filename


def read_wav_info(path: str | Path) -> WavInfo:
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as handle:
        channels = handle.getnchannels()
        frame_rate = handle.getframerate()
        frame_count = handle.getnframes()
    duration = frame_count / frame_rate if frame_rate else 0.0
    return WavInfo(
        path=wav_path,
        sample_rate=frame_rate,
        channels=channels,
        frame_count=frame_count,
        duration_seconds=duration,
    )


def extract_audio(
    media_path: str | Path,
    cache_root: str | Path,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    sample_rate: int = 16_000,
    channels: int = 1,
    overwrite: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    timeout: float = 60.0,
) -> AudioExtractionResult:
    input_path = Path(media_path)
    output_path = audio_cache_path(
        input_path,
        cache_root,
        sample_rate=sample_rate,
        channels=channels,
        extension=".wav",
    )
    if output_path.exists() and not overwrite:
        info = _safe_wav_info(output_path)
        return AudioExtractionResult(
            input_path=input_path,
            output_path=output_path,
            success=True,
            reused=True,
            duration_seconds=info.duration_seconds if info else None,
            sample_rate=info.sample_rate if info else sample_rate,
            channels=info.channels if info else channels,
            source="cache",
        )

    resolved_ffmpeg = resolve_ffmpeg(ffmpeg_bin)
    if resolved_ffmpeg is None:
        return AudioExtractionResult(
            input_path=input_path,
            output_path=output_path,
            success=False,
            error=f"{ffmpeg_bin} not found",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [resolved_ffmpeg, "-y" if overwrite else "-n"]
    if start_seconds is not None:
        command.extend(["-ss", _ffmpeg_time(start_seconds)])
    command.extend(["-i", str(input_path)])
    if duration_seconds is not None:
        command.extend(["-t", _ffmpeg_time(duration_seconds)])
    command.extend(["-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(channels), str(output_path)])

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return AudioExtractionResult(
            input_path=input_path,
            output_path=output_path,
            success=False,
            command=tuple(command),
            error=str(exc),
        )

    if completed.returncode != 0:
        return AudioExtractionResult(
            input_path=input_path,
            output_path=output_path,
            success=False,
            command=tuple(command),
            error=(completed.stderr or completed.stdout).strip() or "ffmpeg failed",
        )

    info = _safe_wav_info(output_path)
    return AudioExtractionResult(
        input_path=input_path,
        output_path=output_path,
        success=True,
        duration_seconds=info.duration_seconds if info else None,
        sample_rate=info.sample_rate if info else sample_rate,
        channels=info.channels if info else channels,
        command=tuple(command),
    )


def _ffmpeg_time(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    return f"{seconds:.3f}"


def _safe_wav_info(path: Path) -> WavInfo | None:
    try:
        return read_wav_info(path)
    except (OSError, EOFError, wave.Error):
        return None
