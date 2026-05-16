from backend.app.embeddings.vector_store import VectorSearchHit
from backend.app.hf_pipeline.retrieval import RetrievalSettings, late_fuse_hits, normalize_scores


def _hit(point_id: str, score: float, clip_id: int, field: str, text: str) -> VectorSearchHit:
    return VectorSearchHit(
        point_id=point_id,
        score=score,
        payload={
            "clip_id": clip_id,
            "file_name": f"clip_{clip_id}.mp4",
            "field": field,
            "window_id": "window_001",
            "start_sec": 0.0,
            "end_sec": 2.0,
            "payload_text": text,
        },
    )


def test_normalize_scores_uses_minmax_per_field() -> None:
    hits = [_hit("a", 0.2, 1, "summary", "low"), _hit("b", 0.6, 2, "summary", "high")]

    assert normalize_scores(hits) == {"a": 0.0, "b": 1.0}
    assert normalize_scores([_hit("only", 0.4, 1, "summary", "same")]) == {"only": 1.0}


def test_late_fusion_merges_candidates_by_clip_and_weights_fields() -> None:
    result = late_fuse_hits(
        "window rotation",
        {
            "summary": [
                _hit("summary-1", 0.9, 1, "summary", "rotation near window"),
                _hit("summary-2", 0.1, 2, "summary", "quiet map"),
            ],
            "speech": [_hit("speech-1", 0.7, 1, "speech", "rotate left")],
            "metadata": [_hit("metadata-2", 0.7, 2, "metadata", "clip_2.mp4")],
            "invalid": [_hit("ignored", 1.0, 3, "invalid", "ignored")],
        },
        settings=RetrievalSettings(rerank_top_n=10),
    )

    assert [candidate.clip_id for candidate in result.candidates] == [1, 2]
    first = result.candidates[0]
    assert first.matched_fields == ["speech", "summary"]
    assert first.combined_score == 0.35
    assert first.field_scores == {"summary": 0.2, "speech": 0.15}
    assert {pointer.source for pointer in first.evidence_pointers} == {"metadata", "speech"}
    assert result.per_field_counts == {"summary": 2, "speech": 1, "metadata": 1}


def test_late_fusion_limits_candidates_before_reranking() -> None:
    hits = [_hit(f"summary-{index}", float(index), index, "summary", f"clip {index}") for index in range(5)]

    result = late_fuse_hits("query", {"summary": hits}, settings=RetrievalSettings(rerank_top_n=2))

    assert len(result.candidates) == 2
    assert result.settings["per_field_top_k"] == 50
    assert result.settings["final_top_k"] == 10
