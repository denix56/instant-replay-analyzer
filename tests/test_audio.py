from __future__ import annotations

import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.processing.audio import audio_cache_path, extract_audio, read_wav_info


def write_silent_wav(path: Path, *, sample_rate: int = 8_000, frames: int = 800) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


def test_read_wav_info(tmp_path: Path) -> None:
    wav_path = tmp_path / "audio.wav"
    write_silent_wav(wav_path)

    info = read_wav_info(wav_path)

    assert info.sample_rate == 8_000
    assert info.channels == 1
    assert info.frame_count == 800
    assert info.duration_seconds == 0.1


def test_extract_audio_reuses_existing_cache_without_ffmpeg(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    cache_root = tmp_path / "cache"
    media.write_bytes(b"fake")
    cached = audio_cache_path(media, cache_root, sample_rate=8_000, channels=1)
    write_silent_wav(cached)

    result = extract_audio(
        media,
        cache_root,
        sample_rate=8_000,
        channels=1,
        ffmpeg_bin="definitely_missing_ffmpeg",
    )

    assert result.success is True
    assert result.reused is True
    assert result.output_path == cached
    assert result.sample_rate == 8_000
    assert result.channels == 1


def test_extract_audio_gracefully_reports_missing_ffmpeg(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")

    result = extract_audio(media, tmp_path / "cache", ffmpeg_bin="definitely_missing_ffmpeg")

    assert result.success is False
    assert result.error == "definitely_missing_ffmpeg not found"
    assert not result.output_path.exists()
