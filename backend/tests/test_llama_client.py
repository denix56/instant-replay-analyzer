import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.llm.llama_client import LlamaClient, LlamaConfig


def test_llama_client_mock_is_deterministic():
    client = LlamaClient()

    assert client.complete("hello world") == "hello world"


def test_llama_client_falls_back_when_endpoint_unavailable():
    client = LlamaClient(LlamaConfig(endpoint="http://127.0.0.1:9/api/generate", timeout_seconds=0.01))

    assert "Question" in client.complete("Question: what happened?")
