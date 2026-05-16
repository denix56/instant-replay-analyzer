from pathlib import Path

import pytest

from backend.app.embeddings.hf_multimodal_embedder import (
    EmbeddingConfig,
    HuggingFaceMultimodalEmbedder,
    TransformersEmbeddingBackend,
)
from backend.app.hf_pipeline.model_registry import model_for_role


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def embed_query_texts(self, texts):  # noqa: ANN001, ANN201 - small test double.
        values = [str(text) for text in texts]
        self.calls.append(("query", values))
        return [[1.0, 0.0] for _ in values]

    def embed_texts(self, texts):  # noqa: ANN001, ANN201 - small test double.
        values = [str(text) for text in texts]
        self.calls.append(("document", values))
        return [[0.0, 1.0] for _ in values]


class RecordingVideoBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def embed_multimodal(self, **kwargs):  # noqa: ANN003, ANN201 - small test double.
        self.calls.append(kwargs)
        return [0.25, 0.75]


def test_embedder_uses_query_and_document_sides() -> None:
    backend = RecordingBackend()
    embedder = HuggingFaceMultimodalEmbedder(EmbeddingConfig(dimension=2), backend=backend)  # type: ignore[arg-type]

    assert embedder.embed_query("where is the Auto-5 kill?") == [1.0, 0.0]
    assert embedder.embed_text("Auto-5 kill near the window") == [0.0, 1.0]
    assert backend.calls == [
        ("query", ["where is the Auto-5 kill?"]),
        ("document", ["Auto-5 kill near the window"]),
    ]


def test_hash_fallback_is_deterministic_and_dimensioned() -> None:
    embedder = HuggingFaceMultimodalEmbedder(EmbeddingConfig(dimension=12))

    first = embedder.embed_text("same payload")
    second = embedder.embed_text("same payload")

    assert first == second
    assert len(first) == 12
    assert any(value != 0 for value in first)


def test_multimodal_embedder_accepts_audio_and_image_paths(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    audio = tmp_path / "audio.wav"
    image.write_bytes(b"fake-image")
    audio.write_bytes(b"fake-audio")

    embedder = HuggingFaceMultimodalEmbedder(EmbeddingConfig(dimension=16))
    vector = embedder.embed_multimodal(text="boss lair fight", image_path=image, audio_path=audio)

    assert len(vector) == 16
    assert any(value != 0 for value in vector)


def test_multimodal_embedder_rejects_frame_paths(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-frame")
    embedder = HuggingFaceMultimodalEmbedder(EmbeddingConfig(dimension=16))

    with pytest.raises(ValueError, match="frame_paths are reserved"):
        embedder.embed_multimodal(text="full clip", frame_paths=[frame])


def test_multimodal_embedder_passes_video_path_and_sampling_to_backend(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    backend = RecordingVideoBackend()
    embedder = HuggingFaceMultimodalEmbedder(
        EmbeddingConfig(dimension=2, video_fps=2.0, video_max_frames=64),
        backend=backend,  # type: ignore[arg-type]
    )

    vector = embedder.embed_multimodal(text="full clip", video_path=video)

    assert vector == [pytest.approx(0.316227766), pytest.approx(0.948683298)]
    assert backend.calls == [
        {
            "text": "full clip",
            "image_path": None,
            "video_path": video,
            "frame_paths": None,
            "video_fps": 2.0,
            "video_max_frames": 64,
            "audio_path": None,
        }
    ]


def test_video_hash_fallback_is_deterministic_and_dimensioned(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    embedder = HuggingFaceMultimodalEmbedder(EmbeddingConfig(dimension=16))

    first = embedder.embed_video_path(video, text="full clip", fps=2.0, max_frames=64)
    second = embedder.embed_video_path(video, text="full clip", fps=2.0, max_frames=64)

    assert first == second
    assert len(first) == 16
    assert any(value != 0 for value in first)


def test_transformers_backend_sends_full_video_as_one_payload(tmp_path: Path) -> None:
    calls: list[object] = []
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")

    class FakeManager:
        def embed(self, spec, values):  # noqa: ANN001, ANN201 - test double.
            calls.append((spec.model_id, values))
            return [[0.25, 0.75]]

    backend = TransformersEmbeddingBackend(
        model_for_role("embedder"),
        manager=FakeManager(),  # type: ignore[arg-type]
        video_fps=2.0,
        video_max_frames=64,
    )

    assert backend.embed_video_path(video, text="full clip") == [0.25, 0.75]
    assert len(calls) == 1
    payload = calls[0][1][0]
    assert payload["video_path"] == str(video)
    assert payload["video_fps"] == 2.0
    assert payload["video_max_frames"] == 64
    assert "frame_paths" not in payload


def test_transformers_backend_rejects_frame_sequence_embedding(tmp_path: Path) -> None:
    frame_1 = tmp_path / "frame_1.jpg"
    frame_2 = tmp_path / "frame_2.jpg"
    frame_1.write_bytes(b"fake-frame-1")
    frame_2.write_bytes(b"fake-frame-2")

    class FakeManager:
        def embed(self, spec, values):  # noqa: ANN001, ANN201 - test double.
            return [[0.5, 0.5]]

    backend = TransformersEmbeddingBackend(
        model_for_role("embedder"),
        manager=FakeManager(),  # type: ignore[arg-type]
        video_fps=2.0,
        video_max_frames=64,
    )

    with pytest.raises(ValueError, match="frame_paths are reserved"):
        backend.embed_multimodal(text="full clip", frame_paths=[frame_1, frame_2])
