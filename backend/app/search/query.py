from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from ..embeddings.hf_multimodal_embedder import HuggingFaceMultimodalEmbedder
from ..embeddings.vector_store import VectorStore, VectorSearchHit, metadata_from_payload
from ..llm.schemas import ClipRecord, SearchRequest, SearchResult
from .ranker import SearchRanker


@dataclass(frozen=True)
class SearchServiceConfig:
    prefetch_multiplier: int = 4


class ClipSearchService:
    def __init__(
        self,
        *,
        embedder: Optional[HuggingFaceMultimodalEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        ranker: Optional[SearchRanker] = None,
        config: Optional[SearchServiceConfig] = None,
    ) -> None:
        self.embedder = embedder or HuggingFaceMultimodalEmbedder()
        self.vector_store = vector_store or VectorStore()
        self.ranker = ranker or SearchRanker()
        self.config = config or SearchServiceConfig()

    def index_clip(self, clip: ClipRecord) -> None:
        self.vector_store.upsert_clip(clip)

    def index_clips(self, clips: Iterable[ClipRecord]) -> None:
        for clip in clips:
            self.index_clip(clip)

    def search(self, request: SearchRequest | str) -> List[SearchResult]:
        search_request = request if isinstance(request, SearchRequest) else SearchRequest(query=request)
        vector = self.embedder.embed_query(search_request.query)
        filters = _filters_from_request(search_request)
        limit = max(search_request.limit, 1)
        hits = self.vector_store.search(
            vector,
            limit=limit * max(1, self.config.prefetch_multiplier),
            filters=filters,
        )
        candidates = [_hit_to_result(hit) for hit in hits]
        ranked = self.ranker.rank(
            search_request,
            candidates,
            vector_scores={hit.point_id: hit.score for hit in hits},
        )
        return ranked[: search_request.limit]


def _filters_from_request(request: SearchRequest) -> dict[str, object]:
    filters: dict[str, object] = {}
    if request.game:
        filters["metadata.game"] = request.game
    if request.tags:
        filters["metadata.tags"] = request.tags
    return filters


def _hit_to_result(hit: VectorSearchHit) -> SearchResult:
    metadata = metadata_from_payload(hit.payload, hit.point_id)
    return SearchResult(
        clip_id=str(hit.payload.get("clip_id") or hit.point_id),
        score=hit.score,
        metadata=metadata,
        transcript=str(hit.payload.get("transcript") or ""),
        summary=str(hit.payload.get("summary") or ""),
        reasons=[f"vector={hit.score:.2f}"],
    )
