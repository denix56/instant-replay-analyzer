from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from ..llm.llama_client import LlamaClient
from ..llm.prompts import DEEP_REASONING_SYSTEM_PROMPT, build_reasoning_prompt
from ..llm.schemas import ReasoningResult, SearchResult


@dataclass(frozen=True)
class DeepReasoningConfig:
    use_llm: bool = False
    mock_fallback: bool = True
    max_evidence: int = 5


class GameplayReasoner:
    def __init__(
        self,
        config: Optional[DeepReasoningConfig] = None,
        llama_client: Optional[LlamaClient] = None,
    ) -> None:
        self.config = config or DeepReasoningConfig()
        self._llama_client = llama_client or LlamaClient()

    def answer(self, question: str, results: Iterable[SearchResult]) -> ReasoningResult:
        evidence = list(results)[: self.config.max_evidence]
        if self.config.use_llm:
            try:
                answer = self._llama_client.complete(
                    build_reasoning_prompt(question, evidence),
                    system=DEEP_REASONING_SYSTEM_PROMPT,
                    temperature=0.0,
                )
                return ReasoningResult(
                    answer=answer,
                    evidence_clip_ids=[result.clip_id for result in evidence],
                    engine="llama",
                )
            except Exception:
                if not self.config.mock_fallback:
                    raise
        return self._mock_answer(question, evidence)

    @staticmethod
    def _mock_answer(question: str, evidence: list[SearchResult]) -> ReasoningResult:
        terms = {term for term in re.findall(r"[a-z0-9_]+", question.lower()) if len(term) > 2}
        selected = []
        for result in evidence:
            text = " ".join([result.summary, result.transcript, result.metadata.title]).lower()
            if not terms or any(term in text for term in terms):
                selected.append(result)
        selected = selected or evidence[:1]
        if not selected:
            return ReasoningResult(answer="No matching clips were provided.", engine="mock")
        ids = [result.clip_id for result in selected]
        answer = "Relevant clips: " + ", ".join(ids)
        return ReasoningResult(answer=answer, evidence_clip_ids=ids, engine="mock")
