import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.llm.schemas import ClipMetadata, SearchRequest, SearchResult
from backend.app.search.ranker import SearchRanker


def test_ranker_combines_vector_and_lexical_scores():
    candidates = [
        SearchResult(
            clip_id="a",
            score=0.4,
            metadata=ClipMetadata(clip_id="a", path="a.mp4", title="quiet rotation"),
            summary="rotating across the map",
        ),
        SearchResult(
            clip_id="b",
            score=0.4,
            metadata=ClipMetadata(clip_id="b", path="b.mp4", title="clutch final kill"),
            summary="player gets the final kill and wins",
        ),
    ]

    ranked = SearchRanker().rank("final kill", candidates)

    assert [result.clip_id for result in ranked] == ["b", "a"]
    assert ranked[0].highlights


def test_ranker_applies_metadata_match():
    candidate = SearchResult(
        clip_id="clip",
        score=0.1,
        metadata=ClipMetadata(
            clip_id="clip",
            path="clip.mp4",
            game="Arena",
            tags=["win", "kill"],
        ),
    )

    ranked = SearchRanker().rank(SearchRequest(query="", game="Arena", tags=["win"]), [candidate])

    assert ranked[0].score > candidate.score
    assert any(reason.startswith("metadata=") for reason in ranked[0].reasons)
