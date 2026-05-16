import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.models import SearchResult
from backend.app.hf_pipeline.model_registry import model_for_role
from backend.app.search.reranking import SearchReranker, _ranked_scores_from_output, _result_document


def _result(clip_id: int, filename: str, summary: str, *, weapon: str | None = None, death_status: str | None = None) -> SearchResult:
    return SearchResult(
        clip_id=clip_id,
        clip_filename=filename,
        source_path=filename,
        group_name="Hunt",
        summary=summary,
        tags=[],
        score=1.0,
        active_weapon=weapon,
        active_equipment=weapon,
        detected_loadout=[weapon] if weapon else [],
        death_status=death_status,
        killed_by_weapon="Hunting Bow" if death_status else None,
    )


def test_transformers_reranker_fallback_prefers_player_kill_intent(tmp_path):
    reranker = SearchReranker(
        model_path=tmp_path / "missing",
        model_name="Qwen/Qwen3-VL-Reranker-2B",
        mock_fallback=True,
    )
    results = [
        _result(1, "Hunt Player downed.DVR.mp4", "You were downed near a window with Auto-5.", weapon="Auto-5", death_status="downed"),
        _result(2, "Hunt 23.23.53.25.Hunter killed.DVR.mp4", "You killed a hunter near a right-side window.", weapon="Auto-5"),
    ]

    output = reranker.rerank("i kill with auto5 near the window", results)

    assert output.used_model is False
    assert output.results[0].clip_id == 2
    assert "Reranked by local fallback" in output.results[0].matched_reason


def test_transformers_reranker_missing_runtime_raises_when_fallback_disabled(tmp_path):
    reranker = SearchReranker(
        model_path=tmp_path / "missing",
        model_name="Qwen/Qwen3-VL-Reranker-2B",
        mock_fallback=False,
    )

    with pytest.raises(RuntimeError, match="Transformers runtime is not configured"):
        reranker.rerank("query", [_result(1, "clip.mp4", "summary")])


def test_transformers_reranker_uses_manager(monkeypatch):
    class FakeManager:
        def rerank(self, spec, query, documents):  # noqa: ANN001, ANN201
            assert spec.model_id == "Qwen/Qwen3-VL-Reranker-2B"
            assert query == "window fight"
            assert len(documents) == 2
            return [0.1, 0.9]

    reranker = SearchReranker(
        model_name="Qwen/Qwen3-VL-Reranker-2B",
        spec=model_for_role("reranker"),
        manager=FakeManager(),  # type: ignore[arg-type]
        mock_fallback=False,
    )

    output = reranker.rerank(
        "window fight",
        [_result(1, "a.mp4", "a"), _result(2, "b.mp4", "b")],
    )

    assert output.used_model is True
    assert output.results[0].clip_id == 2


def test_reranker_aligns_ranked_scores_to_original_documents():
    ranked = [
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.2},
        {"index": 1, "score": 0.5},
    ]

    assert _ranked_scores_from_output(ranked, 3) == {0: 0.2, 1: 0.5, 2: 0.9}


def test_reranker_document_is_structured_for_model_relevance():
    result = _result(3, "clip.mp4", "Enemy visible near window.", weapon="Auto-5")
    result = result.model_copy(update={"matched_reason": "Matched video and speech.", "segment_start": 1.0, "segment_end": 3.5})

    document = _result_document(result)

    assert "file_name: clip.mp4" in document
    assert "summary: Enemy visible near window." in document
    assert "matched_reason: Matched video and speech." in document
    assert "time_window: 1.000-3.500s" in document
    assert "active_weapon: Auto-5" in document
