from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence

from ..hf_pipeline.model_registry import HFModelSpec, VIDEO_MAX_FRAMES_QWEN_DEFAULT, model_config
from ..llm.schemas import Vector
from ..runtime.transformers_runtime import TransformersModelManager


DEFAULT_EMBEDDING_DIMENSION = 2048
DEFAULT_EMBEDDING_MODEL = model_config("embedder").default_model_id


class EmbeddingBackend(Protocol):
    def embed_query_texts(self, texts: Sequence[str]) -> List[Vector]:
        ...

    def embed_texts(self, texts: Sequence[str]) -> List[Vector]:
        ...

    def embed_image_path(self, path: str | Path) -> Vector:
        ...

    def embed_image_paths(self, paths: Sequence[str | Path]) -> List[Vector]:
        ...

    def embed_audio_path(self, path: str | Path) -> Vector:
        ...

    def embed_video_path(
        self,
        path: str | Path,
        *,
        text: str = "",
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> Vector:
        ...

    def embed_video_frames(
        self,
        frame_paths: Sequence[str | Path],
        *,
        text: str = "",
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> Vector:
        ...

    def embed_multimodal(
        self,
        *,
        text: str = "",
        image_path: Optional[str | Path] = None,
        audio_path: Optional[str | Path] = None,
        video_path: Optional[str | Path] = None,
        frame_paths: Optional[Sequence[str | Path]] = None,
        video_fps: float | None = None,
        video_max_frames: int | None = None,
    ) -> Vector:
        ...


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    dimension: int = DEFAULT_EMBEDDING_DIMENSION
    modality: str = "multimodal"
    normalize: bool = True
    mock_fallback: bool = True
    model_path: Optional[Path] = None
    runtime_backend: str = "unknown"
    precision: str = "unknown"
    quantization: str = "configurable"
    video_fps: float = 6.0
    video_max_frames: int = VIDEO_MAX_FRAMES_QWEN_DEFAULT
    video_instruction: str = "Represent the gameplay video content for retrieval."


class HuggingFaceMultimodalEmbedder:
    """Multimodal embedding facade with deterministic fallback for tests."""

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        self.config = config or EmbeddingConfig()
        if self.config.dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        self._backend = backend

    @property
    def dimension(self) -> int:
        return self.config.dimension

    @property
    def uses_real_backend(self) -> bool:
        return self._backend is not None

    def embed_text(self, text: str) -> Vector:
        return self.embed_texts([text])[0]

    def embed_query(self, text: str) -> Vector:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: Sequence[str]) -> List[Vector]:
        if self._backend is None and not self.config.mock_fallback:
            raise RuntimeError(
                "Real Transformers embedding backend is not configured. Install model weights "
                "or enable mock fallback for deterministic tests."
            )
        if self._backend is not None and hasattr(self._backend, "embed_query_texts"):
            try:
                vectors = self._backend.embed_query_texts(texts)
                return [self._fit_dimension(vector) for vector in vectors]
            except Exception:
                if not self.config.mock_fallback:
                    raise
        return [self._hash_text(f"query: {text}") for text in texts]

    def embed_texts(self, texts: Sequence[str]) -> List[Vector]:
        if self._backend is None and not self.config.mock_fallback:
            raise RuntimeError(
                "Real Transformers embedding backend is not configured. Install model weights "
                "or enable mock fallback for deterministic tests."
            )
        if self._backend is not None:
            try:
                vectors = self._backend.embed_texts(texts)
                return [self._fit_dimension(vector) for vector in vectors]
            except Exception:
                if not self.config.mock_fallback:
                    raise
        return [self._hash_text(f"document: {text}") for text in texts]

    def embed_image_bytes(self, image: bytes, *, hint: str = "") -> Vector:
        if self._backend is None and not self.config.mock_fallback:
            raise RuntimeError("Real Hugging Face image embedding backend is not configured.")
        return self._hash_bytes(hint.encode("utf-8") + b"\0" + image)

    def embed_image_path(self, path: str | Path) -> Vector:
        if self._backend is not None and hasattr(self._backend, "embed_image_path"):
            try:
                return self._fit_dimension(self._backend.embed_image_path(path))
            except Exception:
                if not self.config.mock_fallback:
                    raise
        file_path = Path(path)
        try:
            return self.embed_image_bytes(file_path.read_bytes(), hint=file_path.name)
        except OSError:
            if not self.config.mock_fallback:
                raise
            return self._hash_text(file_path.stem)

    def embed_image_paths(self, paths: Sequence[str | Path]) -> List[Vector]:
        if self._backend is not None and hasattr(self._backend, "embed_image_paths"):
            try:
                vectors = self._backend.embed_image_paths(paths)
                return [self._fit_dimension(vector) for vector in vectors]
            except Exception:
                if not self.config.mock_fallback:
                    raise
        return [self.embed_image_path(path) for path in paths]

    def embed_audio_bytes(self, audio: bytes, *, hint: str = "") -> Vector:
        if self._backend is None and not self.config.mock_fallback:
            raise RuntimeError("Real Hugging Face audio embedding backend is not configured.")
        return self._hash_bytes(b"audio\0" + hint.encode("utf-8") + b"\0" + audio)

    def embed_audio_path(self, path: str | Path) -> Vector:
        if self._backend is not None and hasattr(self._backend, "embed_audio_path"):
            try:
                return self._fit_dimension(self._backend.embed_audio_path(path))
            except Exception:
                if not self.config.mock_fallback:
                    raise
        file_path = Path(path)
        try:
            return self.embed_audio_bytes(file_path.read_bytes(), hint=file_path.name)
        except OSError:
            if not self.config.mock_fallback:
                raise
            return self._hash_text(file_path.stem)

    def embed_video_path(
        self,
        path: str | Path,
        *,
        text: str = "",
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> Vector:
        selected_fps = fps if fps is not None else self.config.video_fps
        selected_max_frames = max_frames if max_frames is not None else self.config.video_max_frames
        if self._backend is not None and hasattr(self._backend, "embed_video_path"):
            try:
                return self._fit_dimension(
                    self._backend.embed_video_path(
                        path,
                        text=text,
                        fps=selected_fps,
                        max_frames=selected_max_frames,
                    )
                )
            except Exception:
                if not self.config.mock_fallback:
                    raise
        file_path = Path(path)
        hint = (
            f"video:{file_path.name}|fps={selected_fps}|max_frames={selected_max_frames}|"
            f"text={text}"
        )
        try:
            return self._hash_file(file_path, hint=hint)
        except OSError:
            if not self.config.mock_fallback:
                raise
            return self._hash_text(hint)

    def embed_video_frames(
        self,
        frame_paths: Sequence[str | Path],
        *,
        text: str = "",
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> Vector:
        selected_fps = fps if fps is not None else self.config.video_fps
        selected_max_frames = max_frames if max_frames is not None else self.config.video_max_frames
        if self._backend is not None and hasattr(self._backend, "embed_video_frames"):
            try:
                return self._fit_dimension(
                    self._backend.embed_video_frames(
                        frame_paths,
                        text=text,
                        fps=selected_fps,
                        max_frames=selected_max_frames,
                    )
                )
            except Exception:
                if not self.config.mock_fallback:
                    raise
        hint = f"video_frames|fps={selected_fps}|max_frames={selected_max_frames}|text={text}"
        vectors = [self.embed_image_path(path) for path in frame_paths[:selected_max_frames]]
        if text:
            vectors.append(self.embed_text(text))
        return self._normalize(_average_vectors(vectors)) if vectors else self._hash_text(hint)

    def embed_multimodal(
        self,
        *,
        text: str = "",
        image_bytes: Optional[bytes] = None,
        image_path: Optional[str | Path] = None,
        video_path: Optional[str | Path] = None,
        frame_paths: Optional[Sequence[str | Path]] = None,
        video_fps: float | None = None,
        video_max_frames: int | None = None,
        audio_path: Optional[str | Path] = None,
        audio_bytes: Optional[bytes] = None,
    ) -> Vector:
        if frame_paths:
            raise ValueError("frame_paths are reserved for previews and HUD/death detection, not embeddings")
        if (
            self._backend is not None
            and hasattr(self._backend, "embed_multimodal")
            and image_bytes is None
            and audio_bytes is None
        ):
            try:
                return self._fit_dimension(
                    self._backend.embed_multimodal(
                        text=text,
                        image_path=image_path,
                        video_path=video_path,
                        frame_paths=None,
                        video_fps=video_fps if video_fps is not None else self.config.video_fps,
                        video_max_frames=video_max_frames if video_max_frames is not None else self.config.video_max_frames,
                        audio_path=audio_path,
                    )
                )
            except Exception:
                if not self.config.mock_fallback:
                    raise
        vectors: List[Vector] = []
        if text:
            vectors.append(self.embed_text(text))
        if video_path is not None:
            vectors.append(
                self.embed_video_path(
                    video_path,
                    text=text,
                    fps=video_fps,
                    max_frames=video_max_frames,
                )
            )
        if image_bytes is not None:
            vectors.append(self.embed_image_bytes(image_bytes))
        if image_path is not None:
            vectors.append(self.embed_image_path(image_path))
        if audio_bytes is not None:
            vectors.append(self.embed_audio_bytes(audio_bytes))
        if audio_path is not None:
            vectors.append(self.embed_audio_path(audio_path))
        if not vectors:
            return self._hash_text("")
        merged = [sum(items) / len(vectors) for items in zip(*vectors)]
        return self._normalize(merged) if self.config.normalize else merged

    def _hash_text(self, text: str) -> Vector:
        return self._hash_bytes(text.encode("utf-8"))

    def _hash_bytes(self, data: bytes) -> Vector:
        values: list[float] = []
        counter = 0
        while len(values) < self.config.dimension:
            digest = hashlib.sha256(data + counter.to_bytes(4, "little")).digest()
            for offset in range(0, len(digest), 4):
                integer = int.from_bytes(digest[offset : offset + 4], "little", signed=False)
                values.append((integer / 2**32) * 2.0 - 1.0)
                if len(values) >= self.config.dimension:
                    break
            counter += 1
        return self._normalize(values) if self.config.normalize else values

    def _hash_file(self, path: Path, *, hint: str) -> Vector:
        digest = hashlib.sha256(hint.encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return self._hash_bytes(digest.digest())

    def _fit_dimension(self, vector: Sequence[float]) -> Vector:
        fitted = [float(value) for value in vector][: self.config.dimension]
        if len(fitted) < self.config.dimension:
            fitted.extend([0.0] * (self.config.dimension - len(fitted)))
        return self._normalize(fitted) if self.config.normalize else fitted

    @staticmethod
    def _normalize(vector: Sequence[float]) -> Vector:
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if norm == 0.0:
            return [0.0 for _ in vector]
        return [float(value) / norm for value in vector]


class TransformersEmbeddingBackend:
    """Qwen3-VL embedding backend using in-process Transformers."""

    def __init__(
        self,
        spec: HFModelSpec,
        *,
        manager: TransformersModelManager,
        batch_size: int = 4,
        video_fps: float = 6.0,
        video_max_frames: int = VIDEO_MAX_FRAMES_QWEN_DEFAULT,
        video_instruction: str = "Represent the gameplay video content for retrieval.",
    ) -> None:
        self.spec = spec
        self.manager = manager
        self.batch_size = max(1, batch_size)
        self.video_fps = video_fps
        self.video_max_frames = video_max_frames
        self.video_instruction = video_instruction

    def embed_query_texts(self, texts: Sequence[str]) -> List[Vector]:
        return self._embed([f"query: {text}" for text in texts])

    def embed_texts(self, texts: Sequence[str]) -> List[Vector]:
        return self._embed([f"document: {text}" for text in texts])

    def embed_image_path(self, path: str | Path) -> Vector:
        return self._embed([{"image_path": str(path)}])[0]

    def embed_image_paths(self, paths: Sequence[str | Path]) -> List[Vector]:
        return self._embed([{"image_path": str(path)} for path in paths])

    def embed_audio_path(self, path: str | Path) -> Vector:
        return self._embed([{"audio_path": str(path)}])[0]

    def embed_video_path(
        self,
        path: str | Path,
        *,
        text: str = "",
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> Vector:
        selected_fps = fps if fps is not None else self.video_fps
        selected_max_frames = max_frames if max_frames is not None else self.video_max_frames
        return self._embed(
            [
                {
                    "text": text,
                    "instruction": self.video_instruction,
                    "video_path": str(path),
                    "video_fps": selected_fps,
                    "video_max_frames": selected_max_frames,
                }
            ]
        )[0]

    def embed_video_frames(
        self,
        frame_paths: Sequence[str | Path],
        *,
        text: str = "",
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> Vector:
        selected_fps = fps if fps is not None else self.video_fps
        selected_max_frames = max_frames if max_frames is not None else self.video_max_frames
        return self._embed(
            [
                {
                    "text": text,
                    "instruction": self.video_instruction,
                    "video_frames": [str(path) for path in frame_paths[:selected_max_frames]],
                    "video_fps": selected_fps,
                    "video_max_frames": selected_max_frames,
                }
            ]
        )[0]

    def embed_multimodal(
        self,
        *,
        text: str = "",
        image_path: Optional[str | Path] = None,
        audio_path: Optional[str | Path] = None,
        video_path: Optional[str | Path] = None,
        frame_paths: Optional[Sequence[str | Path]] = None,
        video_fps: float | None = None,
        video_max_frames: int | None = None,
    ) -> Vector:
        if frame_paths:
            raise ValueError("frame_paths are reserved for previews and HUD/death detection, not embeddings")
        selected_fps = video_fps if video_fps is not None else self.video_fps
        selected_max_frames = video_max_frames if video_max_frames is not None else self.video_max_frames
        vectors: list[Vector] = []
        if video_path is not None:
            vectors.append(
                self.embed_video_path(
                    video_path,
                    text=text,
                    fps=selected_fps,
                    max_frames=selected_max_frames,
                )
            )

        payload: dict[str, Any] = {"text": text}
        if image_path is not None:
            payload["image_path"] = str(image_path)
        if audio_path is not None:
            payload["text"] = f"{text}\naudio_file_name: {Path(audio_path).name}".strip()
        if image_path is not None or audio_path is not None or (text and not vectors):
            vectors.append(self._embed([payload])[0])
        if not vectors:
            vectors.append(self._embed([text])[0])
        return _average_vectors(vectors)

    def _embed(self, values: Sequence[str | dict[str, Any]]) -> List[Vector]:
        vectors: list[Vector] = []
        text_batch: list[str | dict[str, Any]] = []

        def flush_text_batch() -> None:
            if text_batch:
                batch = list(text_batch)
                text_batch.clear()
                vectors.extend(self.manager.embed(self.spec, batch))

        for value in values:
            prepared = self._server_input(value)
            if _is_multimodal_embedding_item(prepared):
                flush_text_batch()
                vectors.extend(self.manager.embed(self.spec, [prepared]))
                continue
            text_batch.append(prepared)
            if len(text_batch) >= self.batch_size:
                flush_text_batch()
        flush_text_batch()
        return vectors

    def _server_input(self, value: str | dict[str, Any]) -> str | dict[str, Any]:
        if isinstance(value, str):
            return value
        return value


def _average_vectors(vectors: Sequence[Sequence[float]]) -> Vector:
    if not vectors:
        return []
    width = min(len(vector) for vector in vectors)
    if width == 0:
        return []
    return [
        sum(float(vector[index]) for vector in vectors) / len(vectors)
        for index in range(width)
    ]


def _is_multimodal_embedding_item(value: str | dict[str, Any]) -> bool:
    return isinstance(value, dict) and any(value.get(key) for key in ("image_path", "video_path", "video_frames"))
