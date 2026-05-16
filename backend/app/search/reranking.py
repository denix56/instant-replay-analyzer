from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..hf_pipeline.model_registry import HFModelSpec
from ..models import SearchResult
from ..runtime.transformers_runtime import TransformersModelManager


@dataclass(frozen=True)
class RerankOutput:
    results: list[SearchResult]
    source: str
    used_model: bool
    warning: str | None = None


class SearchReranker:
    """Always-on reranker with a Qwen3-VL Transformers backend."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        model_name: str,
        spec: HFModelSpec | None = None,
        manager: TransformersModelManager | None = None,
        runtime_backend: str = "cpu",
        precision: str = "fp32",
        max_length: int = 2048,
        batch_size: int = 2,
        mock_fallback: bool = True,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.model_name = model_name
        self.spec = spec
        self.manager = manager
        self.runtime_backend = runtime_backend
        self.precision = precision
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.mock_fallback = mock_fallback
        self._backend: _TransformersRerankerBackend | None = None

    def rerank(self, query: str, results: Sequence[SearchResult]) -> RerankOutput:
        candidates = list(results)
        if not candidates:
            return RerankOutput(results=[], source="none", used_model=False)
        try:
            scores = self._model_scores(query, candidates)
        except Exception as exc:  # noqa: BLE001 - search must degrade if the local reranker is unavailable.
            if not self.mock_fallback:
                raise
            scores = None
            warning = f"Local Qwen3-VL reranker unavailable; used local fallback: {exc}"
        else:
            warning = None
        if scores is not None:
            return RerankOutput(
                results=_apply_scores(candidates, scores, source=self.model_name, model_based=True),
                source=self.model_name,
                used_model=True,
            )
        fallback_scores = [_local_score(query, result) for result in candidates]
        return RerankOutput(
            results=_apply_scores(candidates, fallback_scores, source="local-intent-reranker", model_based=False),
            source="local-intent-reranker",
            used_model=False,
            warning=warning,
        )

    def _model_scores(self, query: str, results: list[SearchResult]) -> list[float] | None:
        if self.model_name == "disabled":
            raise RuntimeError("reranker model is disabled")
        if self.spec is None or self.manager is None:
            raise RuntimeError("reranker Transformers runtime is not configured")
        backend = self._backend
        if backend is None:
            backend = _TransformersRerankerBackend(
                self.spec,
                self.manager,
                model_name=self.model_name,
                batch_size=self.batch_size,
            )
            self._backend = backend
        return backend.score(query, [_result_document(result) for result in results])


class _TransformersRerankerBackend:
    def __init__(
        self,
        spec: HFModelSpec,
        manager: TransformersModelManager,
        *,
        model_name: str,
        batch_size: int,
    ) -> None:
        self.spec = spec
        self.manager = manager
        self.model_name = model_name
        self.batch_size = batch_size

    def score(self, query: str, documents: list[str]) -> list[float]:
        scores: list[float] = [0.0] * len(documents)
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            predicted = self.manager.rerank(self.spec, query, batch)
            for index, score in enumerate(_scores_to_list(predicted, len(batch))):
                scores[start + index] = score
        return scores


def _ranked_scores_from_output(ranked: Any, count: int) -> dict[int, float]:
    output: dict[int, float] = {}
    for fallback_index, row in enumerate(list(ranked or [])):
        index = _ranked_value(row, "index", fallback_index)
        score = _ranked_value(row, "relevance_score", _ranked_value(row, "score", 0.0))
        try:
            numeric_index = int(index)
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        if 0 <= numeric_index < count:
            output[numeric_index] = numeric_score
    return output


def _scores_to_list(values: Any, count: int) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (int, float)):
        values = [float(values)]
    output = [0.0] * count
    for index, value in enumerate(list(values or [])[:count]):
        if isinstance(value, (list, tuple)):
            value = value[-1] if value else 0.0
        try:
            output[index] = float(value)
        except (TypeError, ValueError):
            output[index] = 0.0
    return output


def _ranked_value(row: Any, key: str, default: Any) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _apply_scores(
    results: list[SearchResult],
    rerank_scores: list[float],
    *,
    source: str,
    model_based: bool,
) -> list[SearchResult]:
    scored: list[tuple[float, SearchResult]] = []
    max_score = max((abs(result.score) for result in results), default=1.0) or 1.0
    normalized_scores = _normalize_scores(rerank_scores)
    for index, result in enumerate(results):
        rerank_score = normalized_scores[index] if index < len(normalized_scores) else 0.0
        base_score = result.score / max_score
        final_score = (0.35 * base_score) + (0.65 * rerank_score)
        reason = _append_rerank_reason(result.matched_reason, source, model_based, rerank_score)
        scored.append((final_score, _copy_result(result, score=round(final_score, 6), matched_reason=reason)))
    return [result for _, result in sorted(scored, key=lambda item: item[0], reverse=True)]


def _normalize_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    finite = [score for score in scores if math.isfinite(score)]
    if not finite:
        return [0.0 for _ in scores]
    low = min(finite)
    high = max(finite)
    if high == low:
        return [0.5 for _ in scores]
    return [(score - low) / (high - low) if math.isfinite(score) else 0.0 for score in scores]


def _local_score(query: str, result: SearchResult) -> float:
    query_terms = _terms(query)
    document = _result_document(result)
    doc_terms = _terms(document)
    overlap = len(query_terms & doc_terms) / max(math.sqrt(len(query_terms) * len(doc_terms)), 1.0)
    score = overlap
    normalized = query.lower()
    compact_query = _compact(query)
    compact_loadout = _compact(" ".join([result.active_weapon or "", result.active_equipment or "", *result.detected_loadout]))
    player_kill_intent = _player_kill_intent(normalized)
    death_intent = _death_intent(normalized)
    if player_kill_intent:
        if _is_player_kill_result(result):
            score += 3.0
        if result.death_status or result.killed_by_weapon:
            score -= 2.0
    if death_intent and (result.death_status or result.killed_by_weapon):
        score += 2.0
    if "window" in normalized and "window" in document.lower():
        score += 0.8
    if compact_loadout and any(token in compact_query for token in _weapon_tokens(compact_loadout)):
        score += 0.8
    if "auto5" in compact_query and "auto5" in compact_loadout:
        score += 0.8
    return score


def _result_document(result: SearchResult) -> str:
    fields = [
        ("file_name", result.clip_filename),
        ("group", result.group_name),
        ("summary", result.summary or ""),
        ("tags", ", ".join(result.tags)),
        ("speech", result.transcript_snippet or ""),
        ("matched_fields", result.matched_modality),
        ("matched_reason", result.matched_reason),
        ("time_window", _time_window(result)),
        ("active_weapon", result.active_weapon or ""),
        ("active_equipment", result.active_equipment or ""),
        ("detected_loadout", ", ".join(result.detected_loadout)),
        ("death_status", result.death_status or ""),
        ("killed_by_weapon", result.killed_by_weapon or ""),
        ("killer_name", result.killer_name or ""),
    ]
    return "\n".join(f"{label}: {value}" for label, value in fields if str(value).strip())


def _time_window(result: SearchResult) -> str:
    if result.segment_start is not None and result.segment_end is not None:
        return f"{result.segment_start:.3f}-{result.segment_end:.3f}s"
    if result.best_timestamp is not None:
        return f"{result.best_timestamp:.3f}s"
    return ""


def _copy_result(result: SearchResult, **updates: Any) -> SearchResult:
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result.__dict__)
    data.update(updates)
    return SearchResult(**data)


def _append_rerank_reason(reason: str, source: str, model_based: bool, score: float) -> str:
    label = source if model_based else "local fallback"
    suffix = f"Reranked by {label} (score={score:.2f})."
    reason = reason.strip()
    return f"{reason} {suffix}".strip()


def _terms(value: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", value.lower()) if len(term) > 2}


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _weapon_tokens(compact_loadout: str) -> set[str]:
    tokens = {compact_loadout}
    match = re.fullmatch(r"([a-z]+)([0-9]+)", compact_loadout)
    if match:
        tokens.add(f"{match.group(1)}{match.group(2)}")
    return tokens


def _death_intent(query: str) -> bool:
    return bool(re.search(r"\b(killed me|killed by|what killed|who killed|downed me|death|dead)\b", query))


def _player_kill_intent(query: str) -> bool:
    if _death_intent(query):
        return False
    return bool(re.search(r"\b(i|we)\s+(kill|killed|down|downed|shot)|hunter killed|enemy killed|final kill\b", query))


def _is_player_kill_result(result: SearchResult) -> bool:
    filename = result.clip_filename.lower()
    return "hunter killed" in filename or "enemy killed" in filename or "hunter_killed" in filename
