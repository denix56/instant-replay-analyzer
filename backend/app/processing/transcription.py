from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from ..hf_pipeline.model_registry import HFModelSpec
from ..llm.schemas import TranscriptionResult, TranscriptionSegment
from ..runtime.transformers_runtime import TransformersModelManager


class ASRBackend(Protocol):
    def transcribe(self, audio_path: str | Path, language: str = "auto") -> TranscriptionResult:
        ...


@dataclass(frozen=True)
class TranscriptionConfig:
    engine: str = "mock"
    language: str = "auto"
    sidecar_suffix: str = ".transcript.txt"
    mock_fallback: bool = True


class Transcriber:
    """ASR facade with sidecar and deterministic filename fallback."""

    def __init__(
        self,
        config: Optional[TranscriptionConfig] = None,
        backend: Optional[ASRBackend] = None,
    ) -> None:
        self.config = config or TranscriptionConfig()
        self._backend = backend

    def transcribe(self, audio_path: str | Path, *, language: Optional[str] = None) -> TranscriptionResult:
        selected_language = language or self.config.language
        if self._backend is None and not self.config.mock_fallback:
            raise RuntimeError(
                "Real Transformers ASR backend is not configured. Configure a local Transformers runtime "
                "or set ALLOW_MOCK_MODELS=true for deterministic fallback tests."
            )
        if self._backend is not None:
            try:
                return self._backend.transcribe(audio_path, selected_language)
            except Exception:
                if not self.config.mock_fallback:
                    raise

        path = Path(audio_path)
        sidecar = self._sidecar_path(path)
        if sidecar.exists():
            text = sidecar.read_text(encoding="utf-8").strip()
            engine = "sidecar"
        else:
            text = self._mock_text_from_path(path)
            engine = self.config.engine if self.config.engine != "auto" else "mock"
        return TranscriptionResult(
            text=text,
            segments=_segments_from_text(text),
            language=selected_language,
            engine=engine,
            confidence=1.0 if text else 0.0,
        )

    def _sidecar_path(self, audio_path: Path) -> Path:
        return audio_path.with_name(audio_path.name + self.config.sidecar_suffix)

    @staticmethod
    def _mock_text_from_path(path: Path) -> str:
        stem = re.sub(r"[_\-]+", " ", path.stem).strip()
        tokens = [token for token in stem.split() if token]
        if not tokens:
            return ""
        return " ".join(tokens)


def transcribe_clip(audio_path: str | Path, config: Optional[TranscriptionConfig] = None) -> TranscriptionResult:
    return Transcriber(config).transcribe(audio_path)


def _segments_from_text(text: str) -> list[TranscriptionSegment]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences and text:
        sentences = [text]
    segments = []
    cursor = 0.0
    for sentence in sentences:
        duration = max(1.0, min(12.0, len(sentence.split()) * 0.45))
        segments.append(
            TranscriptionSegment(
                start=round(cursor, 3),
                end=round(cursor + duration, 3),
                text=sentence,
                confidence=1.0,
            )
        )
        cursor += duration
    return segments


class TransformersASRBackend:
    """Local Whisper ASR runtime using in-process Transformers."""

    def __init__(
        self,
        spec: HFModelSpec,
        *,
        manager: TransformersModelManager,
    ) -> None:
        self.spec = spec
        self.manager = manager

    def transcribe(self, audio_path: str | Path, language: str = "auto") -> TranscriptionResult:
        return self.manager.transcribe(self.spec, audio_path, language=language)


def _whisper_language(language: str) -> str | None:
    normalized = language.strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    return {
        "en": "english",
        "eng": "english",
        "english": "english",
        "de": "german",
        "german": "german",
        "zh": "chinese",
        "chinese": "chinese",
    }.get(normalized, normalized)
