from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_TARGETS = [
    ROOT / "backend" / "app",
    ROOT / "backend" / "requirements.txt",
    ROOT / "backend" / "pyproject.toml",
    ROOT / "pyproject.toml",
    ROOT / ".env.example",
    ROOT / "README.md",
    ROOT / "docs",
    ROOT / "native-ui" / "src-tauri",
]


def _text_files(path: Path):
    if path.is_file():
        yield path
        return
    for child in path.rglob("*"):
        if child.is_file() and "__pycache__" not in child.parts:
            yield child


def test_removed_embedding_module_is_gone() -> None:
    assert not (ROOT / "backend" / "app" / "embeddings" / "jina_omni_embedder.py").exists()


def test_no_production_jina_references_remain() -> None:
    matches: list[str] = []
    for target in PRODUCTION_TARGETS:
        if not target.exists():
            continue
        for path in _text_files(target):
            try:
                text = path.read_text(encoding="utf-8").lower()
            except UnicodeDecodeError:
                continue
            if "jina" in text or "jinaai" in text:
                matches.append(str(path.relative_to(ROOT)))

    assert matches == []


@pytest.mark.parametrize(
    "path",
    [
        ROOT / "models" / "embeddings" / "jina-embeddings-v5-omni-small",
        ROOT / "data" / "models" / "embeddings" / "jina-embeddings-v5-omni-small",
        ROOT / "data" / "packs" / "hunt-knowledge-pack" / "hud_reference_jina_embeddings.jsonl",
        ROOT / "data" / "packs" / "hunt-knowledge-pack" / "hud_reference_jina_embeddings.npz",
    ],
)
def test_stale_jina_artifacts_are_removed(path: Path) -> None:
    assert not path.exists()
