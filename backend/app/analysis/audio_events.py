from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from ..llm.schemas import AudioEvent, AudioEventResult


class AudioEventBackend(Protocol):
    def detect(self, audio_path: str | Path) -> AudioEventResult:
        ...


@dataclass(frozen=True)
class AudioEventConfig:
    mock_fallback: bool = True
    loud_frame_threshold: int = 12_000


class AudioEventDetector:
    def __init__(
        self,
        config: Optional[AudioEventConfig] = None,
        backend: Optional[AudioEventBackend] = None,
    ) -> None:
        self.config = config or AudioEventConfig()
        self._backend = backend

    def detect(self, audio_path: str | Path) -> AudioEventResult:
        if self._backend is not None:
            try:
                return self._backend.detect(audio_path)
            except Exception:
                if not self.config.mock_fallback:
                    raise
        path = Path(audio_path)
        if path.suffix.lower() == ".wav" and path.exists():
            result = self._detect_wav_loud_sections(path)
            if result.events:
                return result
        return AudioEventResult(events=_events_from_filename(path.stem), engine="rules")

    def _detect_wav_loud_sections(self, path: Path) -> AudioEventResult:
        events: list[AudioEvent] = []
        try:
            with wave.open(str(path), "rb") as wav:
                frame_rate = wav.getframerate() or 1
                sample_width = wav.getsampwidth()
                total_frames = wav.getnframes()
                window = max(1, frame_rate // 4)
                for start in range(0, total_frames, window):
                    frames = wav.readframes(window)
                    if _max_sample(frames, sample_width) >= self.config.loud_frame_threshold:
                        start_seconds = start / frame_rate
                        events.append(
                            AudioEvent(
                                name="loud-event",
                                start=round(start_seconds, 3),
                                end=round(start_seconds + window / frame_rate, 3),
                                confidence=0.65,
                            )
                        )
        except (OSError, wave.Error):
            if not self.config.mock_fallback:
                raise
        return AudioEventResult(events=events, engine="wav-rules")


def detect_audio_events(audio_path: str | Path) -> AudioEventResult:
    return AudioEventDetector().detect(audio_path)


def _events_from_filename(stem: str) -> list[AudioEvent]:
    events = []
    rules = {
        "gunshot": ("gunshot", "shots", "shooting"),
        "explosion": ("explosion", "grenade", "boom"),
        "footstep": ("footstep", "steps"),
        "reload": ("reload", "reloading"),
    }
    normalized = re.sub(r"[_\-]+", " ", stem.lower())
    for name, keywords in rules.items():
        if any(keyword in normalized for keyword in keywords):
            events.append(AudioEvent(name=name, start=0.0, end=1.0, confidence=0.7))
    return events


def _max_sample(frames: bytes, sample_width: int) -> int:
    if sample_width <= 0:
        return 0
    max_value = 0
    for offset in range(0, len(frames), sample_width):
        chunk = frames[offset : offset + sample_width]
        if len(chunk) != sample_width:
            continue
        value = int.from_bytes(chunk, byteorder="little", signed=True)
        max_value = max(max_value, abs(value))
    return max_value
