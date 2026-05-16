from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .hf_multimodal_embedder import DEFAULT_EMBEDDING_DIMENSION
from ..llm.schemas import ClipMetadata, ClipRecord, Payload, Vector


@dataclass(frozen=True)
class VectorStoreConfig:
    collection_name: str = "gameplay_clips"
    dimension: int = DEFAULT_EMBEDDING_DIMENSION
    qdrant_url: Optional[str] = None
    prefer_qdrant: bool = True
    mock_fallback: bool = True


@dataclass(frozen=True)
class VectorStorePoint:
    point_id: str
    vector: Vector
    payload: Payload


@dataclass(frozen=True)
class VectorSearchHit:
    point_id: str
    score: float
    payload: Payload


class VectorStore:
    """Qdrant-backed vector store with in-memory fallback."""

    def __init__(self, config: Optional[VectorStoreConfig] = None) -> None:
        self.config = config or VectorStoreConfig()
        self._memory: Dict[str, VectorStorePoint] = {}
        self._client: Any = None
        if self.config.qdrant_url and self.config.prefer_qdrant:
            self._try_connect_qdrant()

    @property
    def using_qdrant(self) -> bool:
        return self._client is not None

    def upsert(self, point_id: str, vector: Vector, payload: Optional[Payload] = None) -> None:
        self.upsert_many([VectorStorePoint(point_id, self._fit_dimension(vector), payload or {})])

    def upsert_clip(self, clip: ClipRecord) -> None:
        self.upsert(
            clip.clip_id,
            clip.embedding,
            {
                "clip_id": clip.clip_id,
                "metadata": {
                    "clip_id": clip.metadata.clip_id,
                    "path": clip.metadata.path,
                    "title": clip.metadata.title,
                    "game": clip.metadata.game,
                    "created_at": clip.metadata.created_at,
                    "duration_seconds": clip.metadata.duration_seconds,
                    "tags": clip.metadata.tags,
                },
                "transcript": clip.transcript,
                "summary": clip.summary,
                "extra": clip.extra,
            },
        )

    def upsert_many(self, points: Iterable[VectorStorePoint]) -> None:
        fitted = [
            VectorStorePoint(point.point_id, self._fit_dimension(point.vector), dict(point.payload))
            for point in points
        ]
        if self._client is not None:
            try:
                self._qdrant_upsert(fitted)
                return
            except Exception:
                if not self.config.mock_fallback:
                    raise
                self._client = None
        for point in fitted:
            self._memory[point.point_id] = point

    def search(
        self,
        query_vector: Vector,
        *,
        limit: int = 10,
        filters: Optional[Payload] = None,
    ) -> List[VectorSearchHit]:
        if limit <= 0:
            return []
        query = self._fit_dimension(query_vector)
        if self._client is not None:
            try:
                return self._qdrant_search(query, limit, filters or {})
            except Exception:
                if not self.config.mock_fallback:
                    raise
                self._client = None
        hits = []
        for point in self._memory.values():
            if not _payload_matches(point.payload, filters or {}):
                continue
            hits.append(VectorSearchHit(point.point_id, _cosine(query, point.vector), point.payload))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]

    def delete(self, point_id: str) -> None:
        if self._client is not None:
            try:
                self._client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=[point_id],
                )
            except Exception:
                if not self.config.mock_fallback:
                    raise
                self._client = None
        self._memory.pop(point_id, None)

    def clear_memory(self) -> None:
        self._memory.clear()

    def _try_connect_qdrant(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams

            client = QdrantClient(url=self.config.qdrant_url)
            collections = client.get_collections().collections
            names = {collection.name for collection in collections}
            if self.config.collection_name in names:
                size = _qdrant_existing_collection_vector_size(client, self.config.collection_name)
                if size is not None and size != self.config.dimension:
                    if not _is_local_qdrant(self.config.qdrant_url):
                        raise RuntimeError(
                            f"Qdrant collection {self.config.collection_name} has vector size {size}; "
                            f"expected {self.config.dimension}."
                        )
                    client.delete_collection(collection_name=self.config.collection_name)
                    names.remove(self.config.collection_name)
            if self.config.collection_name not in names:
                client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(size=self.config.dimension, distance=Distance.COSINE),
                )
            self._client = client
        except Exception:
            if not self.config.mock_fallback:
                raise
            self._client = None

    def _qdrant_upsert(self, points: List[VectorStorePoint]) -> None:
        from qdrant_client.http.models import PointStruct

        self._client.upsert(
            collection_name=self.config.collection_name,
            points=[
                PointStruct(id=point.point_id, vector=point.vector, payload=point.payload)
                for point in points
            ],
        )

    def _qdrant_search(self, query: Vector, limit: int, filters: Payload) -> List[VectorSearchHit]:
        qdrant_filter = _to_qdrant_filter(filters) if filters else None
        raw_hits = _qdrant_nearest(
            self._client,
            collection_name=self.config.collection_name,
            query_vector=query,
            limit=limit,
            query_filter=qdrant_filter,
        )
        return [
            VectorSearchHit(str(hit.id), float(hit.score), dict(hit.payload or {}))
            for hit in raw_hits
        ]

    def _fit_dimension(self, vector: Vector) -> Vector:
        fitted = [float(value) for value in vector][: self.config.dimension]
        if len(fitted) < self.config.dimension:
            fitted.extend([0.0] * (self.config.dimension - len(fitted)))
        return fitted


def metadata_from_payload(payload: Payload, fallback_id: str) -> ClipMetadata:
    raw = payload.get("metadata") or {}
    return ClipMetadata(
        clip_id=str(raw.get("clip_id") or payload.get("clip_id") or fallback_id),
        path=str(raw.get("path") or ""),
        title=str(raw.get("title") or ""),
        game=str(raw.get("game") or ""),
        created_at=raw.get("created_at"),
        duration_seconds=raw.get("duration_seconds"),
        tags=list(raw.get("tags") or []),
    )


def _payload_matches(payload: Payload, filters: Payload) -> bool:
    for key, expected in filters.items():
        actual: Any = payload
        for part in key.split("."):
            if not isinstance(actual, dict) or part not in actual:
                return False
            actual = actual[part]
        if isinstance(expected, list):
            actual_values = actual if isinstance(actual, list) else [actual]
            if not all(item in actual_values for item in expected):
                return False
        elif actual != expected:
            return False
    return True


def _cosine(left: Vector, right: Vector) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _to_qdrant_filter(filters: Payload) -> Any:
    from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue

    conditions = []
    for key, expected in filters.items():
        match = MatchAny(any=expected) if isinstance(expected, list) else MatchValue(value=expected)
        conditions.append(FieldCondition(key=key, match=match))
    return Filter(must=conditions)


def _qdrant_nearest(
    client: Any,
    *,
    collection_name: str,
    query_vector: Vector,
    limit: int,
    query_filter: Any = None,
) -> list[Any]:
    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
        )
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )
    return list(getattr(response, "points", response))


DEFAULT_COLLECTIONS = ("video", "summary", "speech", "audio_caption", "metadata", "fused")


@dataclass(frozen=True)
class QdrantVectorStoreConfig:
    url: str = "local"
    dimension: int = DEFAULT_EMBEDDING_DIMENSION
    collections: tuple[str, ...] = DEFAULT_COLLECTIONS
    prefer_qdrant: bool = True
    mock_fallback: bool = True
    local_path: str | None = None


class QdrantVectorStore:
    """Required multi-collection vector-store abstraction.

    Qdrant is used when reachable. A deterministic in-memory fallback keeps unit
    tests and offline development usable, while status endpoints can still report
    whether Qdrant is active.
    """

    def __init__(self, config: QdrantVectorStoreConfig | None = None) -> None:
        self.config = config or QdrantVectorStoreConfig()
        self._client: Any = None
        self._memory: dict[str, dict[str, VectorStorePoint]] = {
            collection: {} for collection in self.config.collections
        }
        if self.config.prefer_qdrant:
            self._try_connect_qdrant()

    @property
    def using_qdrant(self) -> bool:
        return self._client is not None

    def add_vector(self, collection: str, id: str, vector: Vector, metadata: Payload) -> None:
        self._ensure_collection(collection)
        point = VectorStorePoint(str(id), self._fit_dimension(vector), dict(metadata))
        point.payload.setdefault("vector_store_key", str(id))
        point.payload.setdefault("collection_kind", collection)
        if self._client is not None:
            try:
                self._client.upsert(
                    collection_name=collection,
                    points=[self._point_struct(point, collection)],
                )
                return
            except Exception:
                if not self.config.mock_fallback:
                    raise
                self._client = None
        self._memory.setdefault(collection, {})[str(id)] = point

    def search(
        self,
        collection: str,
        query_vector: Vector,
        top_k: int = 10,
        filters: Payload | None = None,
    ) -> list[VectorSearchHit]:
        if top_k <= 0:
            return []
        self._ensure_collection(collection)
        query = self._fit_dimension(query_vector)
        if self._client is not None:
            try:
                raw_hits = _qdrant_nearest(
                    self._client,
                    collection_name=collection,
                    query_vector=query,
                    limit=top_k,
                    query_filter=_to_qdrant_filter(filters or {}) if filters else None,
                )
                return [
                    VectorSearchHit(
                        str((hit.payload or {}).get("vector_store_key") or hit.id),
                        float(hit.score),
                        dict(hit.payload or {}),
                    )
                    for hit in raw_hits
                ]
            except Exception:
                if not self.config.mock_fallback:
                    raise
                self._client = None

        hits: list[VectorSearchHit] = []
        for point in self._memory.get(collection, {}).values():
            if not _payload_matches(point.payload, filters or {}):
                continue
            hits.append(VectorSearchHit(point.point_id, _cosine(query, point.vector), point.payload))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    def delete_by_clip_id(self, clip_id: int | str) -> None:
        clip_value = str(clip_id)
        for collection in tuple(self._memory):
            self._memory[collection] = {
                key: point
                for key, point in self._memory[collection].items()
                if str(point.payload.get("clip_id")) != clip_value
            }
        if self._client is None:
            return
        try:
            from qdrant_client.http.models import FieldCondition, Filter, FilterSelector, MatchValue

            qfilter = Filter(must=[FieldCondition(key="clip_id", match=MatchValue(value=int(clip_id)))])
            for collection in self.config.collections:
                self._client.delete(collection_name=collection, points_selector=FilterSelector(filter=qfilter))
        except Exception:
            if not self.config.mock_fallback:
                raise
            self._client = None

    def reset(self) -> None:
        for collection in tuple(self._memory):
            self._memory[collection].clear()
        if self._client is None:
            return
        try:
            for collection in self.config.collections:
                self._client.delete_collection(collection_name=collection)
            self._ensure_qdrant_collections()
        except Exception:
            if not self.config.mock_fallback:
                raise
            self._client = None

    def persist(self) -> None:
        # Qdrant persists through its storage mount. In-memory mode is process-local.
        return None

    def _try_connect_qdrant(self) -> None:
        try:
            from qdrant_client import QdrantClient

            if _is_local_qdrant(self.config.url):
                local_path = Path(self.config.local_path or "./data/qdrant")
                local_path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(local_path))
            else:
                self._client = QdrantClient(url=self.config.url)
            self._ensure_qdrant_collections()
        except Exception:
            if not self.config.mock_fallback:
                raise
            self._client = None

    def _ensure_collection(self, collection: str) -> None:
        if collection not in self._memory:
            self._memory[collection] = {}
        if self._client is not None:
            self._ensure_qdrant_collections((collection,))

    def _ensure_qdrant_collections(self, collections: tuple[str, ...] | None = None) -> None:
        from qdrant_client.http.models import Distance, VectorParams

        selected = collections or self.config.collections
        existing = {item.name for item in self._client.get_collections().collections}
        for collection in selected:
            if collection in existing:
                size = _qdrant_existing_collection_vector_size(self._client, collection)
                if size is not None and size != self.config.dimension:
                    if not _is_local_qdrant(self.config.url):
                        raise RuntimeError(
                            f"Qdrant collection {collection} has vector size {size}; expected {self.config.dimension}."
                        )
                    self._client.delete_collection(collection_name=collection)
                    existing.remove(collection)
            if collection not in existing:
                self._client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=self.config.dimension, distance=Distance.COSINE),
                )

    def _point_struct(self, point: VectorStorePoint, collection: str) -> Any:
        from qdrant_client.http.models import PointStruct

        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}:{point.point_id}"))
        return PointStruct(id=point_uuid, vector=point.vector, payload=point.payload)

    def _fit_dimension(self, vector: Vector) -> Vector:
        fitted = [float(value) for value in vector][: self.config.dimension]
        if len(fitted) < self.config.dimension:
            fitted.extend([0.0] * (self.config.dimension - len(fitted)))
        return fitted


def _is_local_qdrant(url: str | None) -> bool:
    value = (url or "local").strip().lower()
    return value in {"", "local", "embedded", "qdrant-local"} or value.startswith("local:")


def _qdrant_collection_vector_size(info: Any) -> int | None:
    vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
    if isinstance(vectors, dict):
        vectors = next(iter(vectors.values()), None)
    size = getattr(vectors, "size", None)
    return int(size) if size is not None else None


def _qdrant_existing_collection_vector_size(client: Any, collection: str) -> int | None:
    if not hasattr(client, "get_collection"):
        return None
    return _qdrant_collection_vector_size(client.get_collection(collection_name=collection))
