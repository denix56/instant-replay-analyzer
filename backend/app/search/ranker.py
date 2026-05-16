from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from ..llm.schemas import SearchRequest, SearchResult


@dataclass(frozen=True)
class RankingConfig:
    vector_weight: float = 0.72
    lexical_weight: float = 0.2
    metadata_weight: float = 0.08


class SearchRanker:
    def __init__(self, config: RankingConfig | None = None) -> None:
        self.config = config or RankingConfig()

    def rank(
        self,
        request: SearchRequest | str,
        candidates: Iterable[SearchResult],
        *,
        vector_scores: dict[str, float] | None = None,
    ) -> List[SearchResult]:
        search_request = request if isinstance(request, SearchRequest) else SearchRequest(query=request)
        candidate_list = list(candidates)
        query_terms = _terms(search_request.query)
        vector_scores = vector_scores or {candidate.clip_id: candidate.score for candidate in candidate_list}
        reranked = []
        for candidate in candidate_list:
            lexical = _lexical_score(query_terms, candidate)
            metadata = _metadata_score(search_request, candidate)
            vector = _clamp(vector_scores.get(candidate.clip_id, candidate.score))
            score = (
                self.config.vector_weight * vector
                + self.config.lexical_weight * lexical
                + self.config.metadata_weight * metadata
            )
            reasons = list(candidate.reasons)
            if lexical > 0:
                reasons.append(f"lexical={lexical:.2f}")
            if metadata > 0:
                reasons.append(f"metadata={metadata:.2f}")
            reranked.append(
                SearchResult(
                    clip_id=candidate.clip_id,
                    score=round(score, 6),
                    metadata=candidate.metadata,
                    transcript=candidate.transcript,
                    summary=candidate.summary,
                    highlights=_highlights(search_request.query, candidate),
                    reasons=reasons,
                )
            )
        return sorted(reranked, key=lambda result: result.score, reverse=True)


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9_]+", text.lower()) if len(term) > 1}


def _lexical_score(query_terms: set[str], result: SearchResult) -> float:
    if not query_terms:
        return 0.0
    text_terms = _terms(" ".join([result.metadata.title, result.summary, result.transcript]))
    if not text_terms:
        return 0.0
    overlap = len(query_terms & text_terms)
    return overlap / math.sqrt(len(query_terms) * len(text_terms))


def _metadata_score(request: SearchRequest, result: SearchResult) -> float:
    score = 0.0
    checks = 0
    if request.game:
        checks += 1
        score += 1.0 if result.metadata.game.lower() == request.game.lower() else 0.0
    if request.tags:
        checks += 1
        tags = {tag.lower() for tag in result.metadata.tags}
        wanted = {tag.lower() for tag in request.tags}
        score += len(tags & wanted) / len(wanted) if wanted else 0.0
    return score / checks if checks else 0.0


def _highlights(query: str, result: SearchResult) -> List[str]:
    query_terms = _terms(query)
    if not query_terms:
        return []
    snippets: list[str] = []
    for source in [result.summary, result.transcript, result.metadata.title]:
        for sentence in re.split(r"(?<=[.!?])\s+", source):
            if any(term in sentence.lower() for term in query_terms):
                snippets.append(sentence.strip())
                break
        if len(snippets) >= 3:
            break
    return snippets


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, (value + 1.0) / 2.0 if value < 0.0 else value))
