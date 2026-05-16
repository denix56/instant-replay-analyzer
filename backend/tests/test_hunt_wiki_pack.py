import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.app.knowledge.hunt_wiki_pack as hunt_wiki_pack
from backend.app.embeddings.hf_multimodal_embedder import EmbeddingConfig, HuggingFaceMultimodalEmbedder
from backend.app.knowledge.hunt_wiki_pack import (
    HuntWikiPackConfig,
    KnowledgeChunk,
    PackImage,
    WikiEntity,
    build_hunt_knowledge_pack,
    crawl_hunt_wiki,
    normalize_wiki_article_url,
    parse_wiki_page,
    validate_hunt_pack_inputs,
)


WEAPONS_HTML = """
<html><head>
<meta name="description" content="Weapons are the main way to deal damage.">
<meta property="og:title" content="Weapons">
<link rel="canonical" href="https://huntshowdown.wiki.gg/wiki/Weapons">
<script>RLCONF={"wgRevisionId":13332,"wgCategories":["Equipment"]};</script>
</head><body>
<h1 id="firstHeading">Weapons</h1>
<div class="mw-parser-output">
<p>Weapons are bought before the match.</p>
<table><tr><th>Ammo</th><td>Compact Ammo</td></tr></table>
<p><a href="/wiki/Weapons/Dolch_96">Dolch 96</a>
<a href="/wiki/Special:Search">Search</a>
<a href="/wiki/File:Dolch.png">File</a></p>
</div>
</body></html>
"""


DOLCH_HTML = """
<html><head>
<meta name="description" content="The Dolch 96 is a semi-automatic pistol.">
<meta property="og:title" content="Weapons/Dolch 96">
<link rel="canonical" href="https://huntshowdown.wiki.gg/wiki/Weapons/Dolch_96">
<script>RLCONF={"wgRevisionId":14111,"wgCategories":["Weapons","Dolch Ammo"]};</script>
</head><body>
<h1 id="firstHeading">Dolch 96</h1>
<div class="mw-parser-output">
<p>The Dolch 96 uses special ammo and fires quickly.</p>
<p><img
  alt="Weapon Dolch 96.png"
  src="/images/Weapon_Dolch_96.png?abc123"
  data-file-width="512"
  data-file-height="128"
/></p>
<p><img
  alt="Weapon 3D Dolch 96 Bad Blood.jpg"
  src="/images/thumb/Weapon_3D_Dolch_96_Bad_Blood.jpg/700px-Weapon_3D_Dolch_96_Bad_Blood.jpg?skin123"
  data-file-width="1000"
  data-file-height="550"
/></p>
<div class="druid-row"><div class="druid-label">Damage</div><div class="druid-data">97</div></div>
<table><tr><th>Slot</th><td>Small</td></tr><tr><th>Ammo Type</th><td>Special Ammo</td></tr></table>
</div>
</body></html>
"""


AUTO5_SKINS_HTML = """
<html><head>
<meta name="description" content="Crown & King made, semi-automatic shotgun.">
<meta property="og:title" content="Weapons/Auto-5">
<link rel="canonical" href="https://huntshowdown.wiki.gg/wiki/Weapons/Auto-5">
<script>RLCONF={"wgRevisionId":14546,"wgCategories":["Weapons","Weapons/Large Slot"]};</script>
</head><body>
<h1 id="firstHeading">Auto-5</h1>
<div class="mw-parser-output">
<p>Crown & King made, semi-automatic shotgun.</p>
<h2><span class="mw-headline" id="Skins">Skins</span></h2>
<div class="tabber tabber--init">
<header class="tabber__header">
<nav class="tabber__tabs" role="tablist">
<a class="tabber__tab" role="tab" id="Base_Weapon-0-label">Base Weapon</a>
<a class="tabber__tab" role="tab" id="Black_Widow-0-label">Black Widow</a>
<a class="tabber__tab" role="tab" id="No_Clemency-0-label">No Clemency</a>
</nav>
</header>
<section class="tabber__section">
<article class="tabber__panel"><div class="druid-title">No Clemency</div></article>
</section>
</div>
<h2><span class="mw-headline" id="Update_History">Update History</span></h2>
</div>
</body></html>
"""


def test_parse_wiki_page_extracts_content_links_and_key_values():
    page = parse_wiki_page(WEAPONS_HTML, "https://huntshowdown.wiki.gg/wiki/Weapons")

    assert page.title == "Weapons"
    assert page.revision_id == "13332"
    assert page.categories == ["Equipment"]
    assert "Weapons are bought" in page.text
    assert page.key_values["Ammo"] == "Compact Ammo"
    assert page.links == ["https://huntshowdown.wiki.gg/wiki/Weapons/Dolch_96"]


def test_parse_wiki_page_extracts_skin_tabber_names():
    page = parse_wiki_page(AUTO5_SKINS_HTML, "https://huntshowdown.wiki.gg/wiki/Weapons/Auto-5")

    assert page.key_values["Skins"] == "Black Widow, No Clemency"
    assert "No Clemency" in page.text


def test_normalize_wiki_article_url_rejects_disallowed_namespaces_and_queries():
    assert normalize_wiki_article_url("/wiki/Weapons/Dolch_96") == "https://huntshowdown.wiki.gg/wiki/Weapons/Dolch_96"
    assert normalize_wiki_article_url("/wiki/Special:Search") is None
    assert normalize_wiki_article_url("/wiki/File:Dolch.png") is None
    assert normalize_wiki_article_url("/wiki/Weapons?action=edit") is None


def test_crawl_hunt_wiki_skips_missing_pages(tmp_path):
    html = WEAPONS_HTML.replace("/wiki/Weapons/Dolch_96", "/wiki/Weapons/Missing_Thing")

    def fetch(url: str) -> str:
        if url == "https://huntshowdown.wiki.gg/wiki/Weapons":
            return html
        request = httpx.Request("GET", url)
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("not found", request=request, response=response)

    pages = crawl_hunt_wiki(
        HuntWikiPackConfig(
            output_dir=tmp_path / "pack",
            seeds=("Weapons",),
            max_pages=2,
            max_depth=1,
            delay_seconds=0,
            require_real_embeddings=False,
        ),
        fetch_page=fetch,
    )

    assert [page.page_name for page in pages] == ["Weapons"]


def test_crawl_hunt_wiki_does_not_skip_forbidden_pages(tmp_path):
    html = WEAPONS_HTML.replace("/wiki/Weapons/Dolch_96", "/wiki/Weapons/Forbidden_Thing")

    def fetch(url: str) -> str:
        if url == "https://huntshowdown.wiki.gg/wiki/Weapons":
            return html
        request = httpx.Request("GET", url)
        response = httpx.Response(403, request=request, text="Forbidden")
        raise httpx.HTTPStatusError("forbidden", request=request, response=response)

    with pytest.raises(httpx.HTTPStatusError):
        crawl_hunt_wiki(
            HuntWikiPackConfig(
                output_dir=tmp_path / "pack",
                seeds=("Weapons",),
                max_pages=2,
                max_depth=1,
                delay_seconds=0,
                crawl_concurrency=1,
                require_real_embeddings=False,
            ),
            fetch_page=fetch,
        )


def test_build_hunt_knowledge_pack_writes_entities_chunks_and_embeddings(tmp_path):
    pages = {
        "https://huntshowdown.wiki.gg/wiki/Weapons": WEAPONS_HTML,
        "https://huntshowdown.wiki.gg/wiki/Weapons/Dolch_96": DOLCH_HTML,
    }

    def fetch(url: str) -> str:
        return pages[url]

    def fetch_image(url: str) -> bytes:
        assert url in {
            "https://huntshowdown.wiki.gg/images/Weapon_Dolch_96.png?abc123",
            "https://huntshowdown.wiki.gg/images/Weapon_3D_Dolch_96_Bad_Blood.jpg?skin123",
        }
        return b"fake-png"

    output = tmp_path / "pack"
    embedder = HuggingFaceMultimodalEmbedder(EmbeddingConfig(dimension=8))
    manifest = build_hunt_knowledge_pack(
        HuntWikiPackConfig(
            output_dir=output,
            seeds=("Weapons",),
            max_pages=2,
            max_depth=2,
            delay_seconds=0,
            embedding_dimension=8,
            require_real_embeddings=False,
        ),
        embedder=embedder,
        fetch_page=fetch,
        fetch_image=fetch_image,
    )

    assert manifest["page_count"] == 1
    assert manifest["chunk_count"] >= 1
    assert manifest["image_count"] == 2
    assert (output / "crawl_pages.jsonl").exists()
    entities = [json.loads(line) for line in (output / "entities.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(entity["id"] == "weapon:weapons-dolch-96" for entity in entities)
    assert all(entity["type"] in {"weapon", "tool", "consumable", "map"} for entity in entities)
    dolch = next(entity for entity in entities if entity["id"] == "weapon:weapons-dolch-96")
    assert dolch["key_values"]["Damage"] == "97"
    assert dolch["skin_names"] == ["Bad Blood"]
    assert "Bad Blood" in dolch["aliases"]
    assert dolch["image_paths"] == [
        "media/images/weapon/weapon-weapons-dolch-96__0.png",
        "media/images/weapon/weapon-weapons-dolch-96__1.jpg",
    ]
    media = [json.loads(line) for line in (output / "media_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert media[0]["entity_id"] == "weapon:weapons-dolch-96"
    assert (output / media[0]["local_path"]).read_bytes() == b"fake-png"
    vectors = np.load(output / "embeddings.npy")
    assert vectors.shape[0] == manifest["chunk_count"]

    def unexpected_fetch(url: str) -> str:
        raise AssertionError(f"fetch should use crawl cache, got {url}")

    cached_manifest = build_hunt_knowledge_pack(
        HuntWikiPackConfig(
            output_dir=output,
            seeds=("Weapons",),
            max_pages=2,
            max_depth=2,
            delay_seconds=0,
            embedding_dimension=8,
            require_real_embeddings=False,
        ),
        embedder=embedder,
        fetch_page=unexpected_fetch,
        fetch_image=lambda url: (_ for _ in ()).throw(AssertionError(f"image should use cache, got {url}")),
    )

    assert cached_manifest["page_count"] == 1
    assert cached_manifest["image_count"] == 2
    assert vectors.shape[1] == 8


def test_validate_hunt_pack_inputs_fails_before_embedding_work(tmp_path):
    output = tmp_path / "pack"
    output.mkdir()
    page = parse_wiki_page(DOLCH_HTML, "https://huntshowdown.wiki.gg/wiki/Weapons/Dolch_96")
    entity = WikiEntity(
        id="weapon:weapons-dolch-96",
        type="weapon",
        name="Dolch 96",
        aliases=["Dolch 96"],
        description="The Dolch 96 is a semi-automatic pistol.",
        source_url=page.url,
        page_name=page.page_name,
        categories=["Weapons"],
        skin_names=["bad lowercase artifact"],
    )
    chunk = KnowledgeChunk(
        id="weapon:weapons-dolch-96:chunk:0",
        entity_id=entity.id,
        entity_type=entity.type,
        title=entity.name,
        text="The Dolch 96 uses special ammo.",
        source_url=page.url,
        page_name=page.page_name,
        revision_id=page.revision_id,
    )

    with pytest.raises(RuntimeError, match="validation failed before embeddings"):
        validate_hunt_pack_inputs(
            HuntWikiPackConfig(output_dir=output, require_real_embeddings=False),
            [page],
            [entity],
            [chunk],
            [],
        )


def test_build_hunt_knowledge_pack_validates_before_creating_embeddings(tmp_path, monkeypatch):
    pages = {
        "https://huntshowdown.wiki.gg/wiki/Weapons": WEAPONS_HTML,
        "https://huntshowdown.wiki.gg/wiki/Weapons/Dolch_96": DOLCH_HTML,
    }

    def fetch(url: str) -> str:
        return pages[url]

    def fail_validation(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - test double.
        raise RuntimeError("Hunt knowledge pack validation failed before embeddings")

    monkeypatch.setattr(hunt_wiki_pack, "validate_hunt_pack_inputs", fail_validation)

    class FailingEmbedder:
        def embed_texts(self, texts):  # noqa: ANN001, ANN201 - test double.
            raise AssertionError("embedding should not be called for invalid pack data")

    with pytest.raises(RuntimeError, match="validation failed before embeddings"):
        build_hunt_knowledge_pack(
            HuntWikiPackConfig(
                output_dir=tmp_path / "pack",
                seeds=("Weapons",),
                max_pages=2,
                max_depth=2,
                delay_seconds=0,
                embedding_dimension=8,
                require_real_embeddings=False,
                include_images=False,
            ),
            embedder=FailingEmbedder(),  # type: ignore[arg-type]
            fetch_page=fetch,
        )
