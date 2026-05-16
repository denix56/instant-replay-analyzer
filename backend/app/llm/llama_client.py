from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LlamaConfig:
    endpoint: Optional[str] = None
    model: str = "local-llama"
    timeout_seconds: float = 10.0
    mock_fallback: bool = True


class LlamaClient:
    """Small local-LLM facade with deterministic fallback behavior."""

    def __init__(self, config: Optional[LlamaConfig] = None) -> None:
        self.config = config or LlamaConfig()

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        if self.config.endpoint:
            try:
                return self._complete_http(prompt, system, temperature, max_tokens)
            except (OSError, urllib.error.URLError, TimeoutError, ValueError, KeyError):
                if not self.config.mock_fallback:
                    raise
        return self._mock_complete(prompt, system)

    def _complete_http(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.config.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body.get("response") or body.get("content") or body["choices"][0]["text"])

    def _mock_complete(self, prompt: str, system: str = "") -> str:
        text = " ".join((system, prompt)).strip()
        compact = " ".join(text.split())
        if not compact:
            return "No input provided."
        if "Return a short title" in compact or "Return valid JSON only with this schema" in compact:
            first = compact[:80].rstrip()
            return json.dumps(
                {
                    "title": "Gameplay clip",
                    "summary": first,
                    "key_moments": [first] if first else [],
                }
            )
        return compact[:512]
