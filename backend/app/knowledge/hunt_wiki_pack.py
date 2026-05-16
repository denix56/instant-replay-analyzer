from __future__ import annotations

import hashlib
import importlib
import json
import mimetypes
import os
import re
import shutil
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib import robotparser
from urllib.parse import unquote, urldefrag, urljoin, urlparse

import httpx
import numpy as np

from ..config import AppSettings, get_settings
from ..embeddings.hf_multimodal_embedder import (
    DEFAULT_EMBEDDING_DIMENSION,
    EmbeddingConfig,
    HuggingFaceMultimodalEmbedder,
    TransformersEmbeddingBackend,
)
from ..hf_pipeline.model_registry import model_for_role
from ..runtime.transformers_runtime import transformers_runtime_manager
from .hunt_runtime import HF_RETRIEVAL_PROFILE


HUNT_WIKI_BASE_URL = "https://huntshowdown.wiki.gg"
HUNT_WIKI_LICENSE = "Creative Commons Attribution-ShareAlike 4.0 License"
HUNT_WIKI_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0"
PACK_SCHEMA_VERSION = 1

DEFAULT_PACK_ENTITY_TYPES = ("weapon", "tool", "consumable", "map")

DEFAULT_SEED_PAGES = (
    "Weapons",
    "Category:Weapons",
    "Tools",
    "Category:Tools",
    "Consumables",
    "Category:Consumables",
    "Maps",
    "Category:Maps",
)


FetchPage = Callable[[str], str]
FetchBytes = Callable[[str], bytes]


class _ProgressReporter:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.label = ""
        self.total: int | None = None
        self.current = 0
        self.started_at = time.monotonic()
        self.last_emit = 0.0
        self.last_length = 0

    def start(self, label: str, *, total: int | None) -> None:
        self.label = label
        self.total = total
        self.current = 0
        self.started_at = time.monotonic()
        self.last_emit = 0.0
        self._emit(force=True)

    def update(
        self,
        current: int,
        *,
        total: int | None = None,
        detail: str = "",
        force: bool = False,
    ) -> None:
        self.current = current
        if total is not None:
            self.total = total
        now = time.monotonic()
        if force or now - self.last_emit >= 0.2:
            self._emit(detail=detail)

    def finish(self, detail: str = "") -> None:
        if self.total is not None:
            self.current = self.total
        self._emit(detail=detail, force=True)
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self.last_length = 0

    def _emit(self, *, detail: str = "", force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_emit < 0.2:
            return
        self.last_emit = now
        elapsed = max(now - self.started_at, 0.001)
        rate = self.current / elapsed if self.current else 0.0
        if self.total is not None and self.total > 0:
            fraction = min(max(self.current / self.total, 0.0), 1.0)
            filled = int(round(fraction * 24))
            bar = "#" * filled + "-" * (24 - filled)
            status = f"{self.current}/{self.total} {fraction * 100:5.1f}%"
        elif self.total == 0:
            bar = "-" * 24
            status = "0/0 100.0%"
        else:
            bar = "." * 24
            status = f"{self.current}"
        suffix = f" | {detail[:70]}" if detail else ""
        line = f"\r{self.label:<16} [{bar}] {status} {rate:5.1f}/s{suffix}"
        padding = " " * max(self.last_length - len(line), 0)
        sys.stderr.write(line + padding)
        sys.stderr.flush()
        self.last_length = len(line)


@dataclass(frozen=True)
class HuntWikiPackConfig:
    output_dir: Path
    base_url: str = HUNT_WIKI_BASE_URL
    seeds: Sequence[str] = DEFAULT_SEED_PAGES
    allowed_entity_types: Sequence[str] = DEFAULT_PACK_ENTITY_TYPES
    max_pages: int = 750
    max_depth: int = 4
    delay_seconds: float = 0.75
    crawl_concurrency: int = 2
    timeout_seconds: float = 30.0
    batch_size: int = 16
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    require_real_embeddings: bool = True
    include_images: bool = True
    max_images_per_page: int = 12
    min_image_long_side: int = 64
    max_image_bytes: int = 8 * 1024 * 1024
    refresh: bool = False
    progress: bool = False
    reuse_crawl_cache: bool = True
    browser_fetch: bool = False
    selenium_fetch: bool = False
    selenium_remote_url: str = ""
    selenium_headless: bool = True
    selenium_profile_dir: Path | None = None
    show_browser_on_block: bool = False
    user_agent: str = "InstantReplayAnalyzerKnowledgePack/0.1 (+local user initiated)"


@dataclass(frozen=True)
class WikiImage:
    url: str
    alt: str
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class WikiPage:
    title: str
    page_name: str
    url: str
    description: str
    categories: list[str]
    revision_id: str
    text: str
    key_values: dict[str, str]
    links: list[str]
    images: list[WikiImage] = field(default_factory=list)
    crawl_depth: int = 0


@dataclass(frozen=True)
class WikiEntity:
    id: str
    type: str
    name: str
    aliases: list[str]
    description: str
    source_url: str
    page_name: str
    categories: list[str]
    key_values: dict[str, str] = field(default_factory=dict)
    skin_names: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    license: str = HUNT_WIKI_LICENSE
    license_url: str = HUNT_WIKI_LICENSE_URL


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    entity_id: str
    entity_type: str
    title: str
    text: str
    source_url: str
    page_name: str
    revision_id: str
    license: str = HUNT_WIKI_LICENSE
    license_url: str = HUNT_WIKI_LICENSE_URL


@dataclass(frozen=True)
class PackImage:
    id: str
    entity_id: str
    entity_type: str
    title: str
    alt: str
    source_url: str
    page_url: str
    local_path: str
    width: int | None
    height: int | None
    content_type: str
    sha256: str
    bytes: int
    license: str = HUNT_WIKI_LICENSE
    license_url: str = HUNT_WIKI_LICENSE_URL


def build_hunt_knowledge_pack(
    config: HuntWikiPackConfig,
    *,
    embedder: HuggingFaceMultimodalEmbedder | None = None,
    fetch_page: FetchPage | None = None,
    fetch_image: FetchBytes | None = None,
) -> dict[str, object]:
    """Build a redistributable text knowledge pack from allowed Hunt wiki pages.

    The builder intentionally uses normal `/wiki/...` pages instead of `api.php`,
    because wiki.gg's robots rules disallow crawling the API endpoint.
    """

    output_dir = config.output_dir
    if output_dir.exists() and config.refresh:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = _ProgressReporter(enabled=config.progress)
    pages = _load_cached_pages(config)
    browser_fetcher: _BrowserPageFetcher | _SeleniumPageFetcher | None = None
    try:
        if pages and _crawl_cache_is_complete(config, pages):
            progress.start("crawl pages", total=len(pages))
            progress.update(len(pages), detail="cached", force=True)
            progress.finish(f"{len(pages)} cached pages")
        else:
            if pages:
                progress.start("crawl pages", total=config.max_pages if config.max_pages > 0 else None)
                progress.update(len(pages), detail="cached pages", force=True)
            else:
                _reset_crawl_cache(config.output_dir)
                _write_crawl_state(config, [], complete=False)
                progress.start("crawl pages", total=config.max_pages if config.max_pages > 0 else None)
            crawl_config = config
            crawl_fetch_page = fetch_page
            if crawl_fetch_page is None and config.selenium_fetch:
                browser_fetcher = _SeleniumPageFetcher(config)
                crawl_fetch_page = browser_fetcher.fetch
                crawl_config = replace(config, crawl_concurrency=1)
            elif crawl_fetch_page is None and config.browser_fetch:
                browser_fetcher = _BrowserPageFetcher(config)
                crawl_fetch_page = browser_fetcher.fetch
                crawl_config = replace(config, crawl_concurrency=1)
            pages = crawl_hunt_wiki(crawl_config, fetch_page=crawl_fetch_page, progress=progress, existing_pages=pages)
            _write_crawl_state(config, pages, complete=True)
            progress.finish(f"{len(pages)} pages")
    finally:
        if browser_fetcher is not None:
            browser_fetcher.close()

    page_entities = [
        (page, entity)
        for page in pages
        for entity in [_entity_from_page(page)]
        if _entity_allowed(config, entity)
    ]
    pack_pages = [page for page, _entity in page_entities]
    entities = [entity for _page, entity in page_entities]
    images = download_pack_images(config, pack_pages, entities, fetch_image=fetch_image, progress=progress)
    entities = _attach_images_to_entities(entities, images)
    chunks = [
        chunk
        for page, entity in zip(pack_pages, entities)
        for chunk in _chunks_for_page(page, entity)
    ]
    if not chunks:
        raise RuntimeError("No knowledge chunks were produced from the wiki crawl.")

    progress.start("validate pack", total=1)
    validate_hunt_pack_inputs(config, pack_pages, entities, chunks, images)
    progress.update(1, detail=f"{len(entities)} entities, {len(chunks)} chunks")
    progress.finish("ok")

    embedder = embedder or _create_pack_embedder(config)
    progress.start("embed chunks", total=len(chunks))
    vectors = _embed_chunks(embedder, chunks, batch_size=config.batch_size, progress=progress)
    progress.finish(f"{len(chunks)} chunks")

    progress.start("write pack", total=1)
    _write_pack(output_dir, config, pack_pages, entities, chunks, images, vectors)
    progress.update(1, detail="manifest")
    progress.finish(str(output_dir))
    return _manifest(config, pack_pages, entities, chunks, images, vectors)


def crawl_hunt_wiki(
    config: HuntWikiPackConfig,
    *,
    fetch_page: FetchPage | None = None,
    progress: "_ProgressReporter | None" = None,
    existing_pages: Sequence[WikiPage] | None = None,
) -> list[WikiPage]:
    if max(int(config.crawl_concurrency), 1) == 1:
        return _crawl_hunt_wiki_serial(
            config,
            fetch_page=fetch_page,
            progress=progress,
            existing_pages=existing_pages,
        )

    client = _WikiClient(config) if fetch_page is None else None
    fetch = fetch_page or client.fetch  # type: ignore[union-attr]
    seed_urls = [_page_url(config.base_url, seed) for seed in config.seeds]
    pages: list[WikiPage] = list(existing_pages or [])
    seen: set[str] = {page.url for page in pages}
    queued: set[str] = set(seen)
    queue: deque[tuple[str, int]] = deque()
    _seed_resume_queue(config, seed_urls, pages, seen, queued, queue)
    concurrency = max(int(config.crawl_concurrency), 1)
    schedule_delay = max(config.delay_seconds / concurrency, 0.5) if config.delay_seconds > 0 else 0.0
    futures: dict[Future[str], tuple[str, int]] = {}

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            while queue or futures:
                while queue and len(futures) < concurrency and not _page_limit_reached(config, pages):
                    url, depth = queue.popleft()
                    if url in seen:
                        continue
                    seen.add(url)
                    if client is not None and not client.can_fetch(url):
                        continue
                    futures[executor.submit(_fetch_page_with_retries, fetch, url, config)] = (url, depth)
                    if schedule_delay > 0:
                        time.sleep(schedule_delay)

                if not futures:
                    break

                done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    url, depth = futures.pop(future)
                    try:
                        html = future.result()
                    except httpx.HTTPStatusError as exc:
                        if _is_skippable_page_error(exc):
                            if progress is not None:
                                progress.update(
                                    len(pages),
                                    total=config.max_pages if config.max_pages > 0 else None,
                                    detail=f"skip {exc.response.status_code}: {_page_name_from_url(url)}",
                                    force=True,
                                )
                            continue
                        raise
                    page = parse_wiki_page(html, url, base_url=config.base_url)
                    if not page.text and not page.description:
                        continue
                    if _page_limit_reached(config, pages):
                        continue
                    page = replace(page, crawl_depth=depth)
                    pages.append(page)
                    _append_crawl_cache_page(config.output_dir, page)
                    if progress is not None:
                        progress.update(
                            len(pages),
                            total=config.max_pages if config.max_pages > 0 else None,
                            detail=page.title or page.page_name,
                        )
                    if config.max_depth >= 0 and depth >= config.max_depth:
                        continue
                    for link in page.links:
                        normalized = normalize_wiki_article_url(link, config.base_url)
                        if normalized is None or normalized in seen or normalized in queued:
                            continue
                        if not _should_queue_page(config, normalized):
                            continue
                        queued.add(normalized)
                        queue.append((normalized, depth + 1))
    finally:
        if client is not None:
            client.close()
    return pages


def _crawl_hunt_wiki_serial(
    config: HuntWikiPackConfig,
    *,
    fetch_page: FetchPage | None,
    progress: "_ProgressReporter | None",
    existing_pages: Sequence[WikiPage] | None,
) -> list[WikiPage]:
    client = _WikiClient(config) if fetch_page is None else None
    fetch = fetch_page or client.fetch  # type: ignore[union-attr]
    seed_urls = [_page_url(config.base_url, seed) for seed in config.seeds]
    pages: list[WikiPage] = list(existing_pages or [])
    seen: set[str] = {page.url for page in pages}
    queued: set[str] = set(seen)
    queue: deque[tuple[str, int]] = deque()
    _seed_resume_queue(config, seed_urls, pages, seen, queued, queue)

    try:
        while queue and not _page_limit_reached(config, pages):
            url, depth = queue.popleft()
            if url in seen:
                continue
            seen.add(url)
            if client is not None and not client.can_fetch(url):
                continue
            try:
                html = _fetch_page_with_retries(fetch, url, config)
            except httpx.HTTPStatusError as exc:
                if _is_skippable_page_error(exc):
                    if progress is not None:
                        progress.update(
                            len(pages),
                            total=config.max_pages if config.max_pages > 0 else None,
                            detail=f"skip {exc.response.status_code}: {_page_name_from_url(url)}",
                            force=True,
                        )
                    continue
                raise
            page = parse_wiki_page(html, url, base_url=config.base_url)
            if not page.text and not page.description:
                continue
            page = replace(page, crawl_depth=depth)
            pages.append(page)
            _append_crawl_cache_page(config.output_dir, page)
            if progress is not None:
                progress.update(
                    len(pages),
                    total=config.max_pages if config.max_pages > 0 else None,
                    detail=page.title or page.page_name,
                )
            if config.max_depth >= 0 and depth >= config.max_depth:
                continue
            for link in page.links:
                normalized = normalize_wiki_article_url(link, config.base_url)
                if normalized is None or normalized in seen or normalized in queued:
                    continue
                if not _should_queue_page(config, normalized):
                    continue
                queued.add(normalized)
                queue.append((normalized, depth + 1))
            if config.delay_seconds > 0:
                time.sleep(config.delay_seconds)
    finally:
        if client is not None:
            client.close()
    return pages


def _seed_resume_queue(
    config: HuntWikiPackConfig,
    seed_urls: Sequence[str],
    pages: Sequence[WikiPage],
    seen: set[str],
    queued: set[str],
    queue: deque[tuple[str, int]],
) -> None:
    for url in seed_urls:
        if url not in seen and url not in queued:
            queued.add(url)
            queue.append((url, 0))

    for page in pages:
        depth = page.crawl_depth
        if config.max_depth >= 0 and depth >= config.max_depth:
            continue
        for link in page.links:
            normalized = normalize_wiki_article_url(link, config.base_url)
            if normalized is None or normalized in seen or normalized in queued:
                continue
            if not _should_queue_page(config, normalized):
                continue
            queued.add(normalized)
            queue.append((normalized, depth + 1))


def _page_limit_reached(config: HuntWikiPackConfig, pages: Sequence[WikiPage]) -> bool:
    return config.max_pages > 0 and len(pages) >= config.max_pages


def _entity_allowed(config: HuntWikiPackConfig, entity: WikiEntity) -> bool:
    return entity.type in {value.lower() for value in config.allowed_entity_types}


def _should_queue_page(config: HuntWikiPackConfig, url: str) -> bool:
    page_name = _page_name_from_url(url).lower()
    allowed = {value.lower() for value in config.allowed_entity_types}
    allowed_roots = {
        "weapon": ("weapons", "category:weapons"),
        "tool": ("tools", "category:tools"),
        "consumable": ("consumables", "category:consumables"),
        "map": ("maps", "category:maps"),
    }
    for entity_type, roots in allowed_roots.items():
        if entity_type not in allowed:
            continue
        for root in roots:
            if page_name == root or page_name.startswith(root + "/"):
                return True
    return False


def _fetch_page_with_retries(fetch: FetchPage, url: str, config: HuntWikiPackConfig) -> str:
    attempts = 3
    for attempt in range(attempts):
        try:
            return fetch(url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403 and attempt < attempts - 1:
                time.sleep(max(config.delay_seconds * (attempt + 2), 2.0))
                continue
            raise
    raise RuntimeError(f"Unable to fetch {url}")


def download_pack_images(
    config: HuntWikiPackConfig,
    pages: Sequence[WikiPage],
    entities: Sequence[WikiEntity],
    *,
    fetch_image: FetchBytes | None = None,
    progress: "_ProgressReporter | None" = None,
) -> list[PackImage]:
    if not config.include_images:
        if progress is not None:
            progress.start("download images", total=0)
            progress.finish("disabled")
        return []

    media_dir = config.output_dir / "media" / "images"
    media_dir.mkdir(parents=True, exist_ok=True)
    client = _WikiClient(config) if fetch_image is None else None
    images: list[PackImage] = []
    selections = [
        (page, entity, index, image)
        for page, entity in zip(pages, entities)
        for index, image in enumerate(_select_page_images(config, page, entity))
    ]
    if progress is not None:
        progress.start("download images", total=len(selections))

    try:
        for count, (page, entity, index, image) in enumerate(selections, start=1):
            image_id = f"{entity.id}:image:{index}"
            existing = _existing_image_file(config.output_dir, entity, index)
            if existing is not None:
                relative_path = existing.relative_to(config.output_dir)
                source_bytes = existing.read_bytes()
                content_type = mimetypes.guess_type(existing.name)[0] or "application/octet-stream"
                images.append(
                    PackImage(
                        id=image_id,
                        entity_id=entity.id,
                        entity_type=entity.type,
                        title=entity.name,
                        alt=image.alt,
                        source_url=image.url,
                        page_url=page.url,
                        local_path=relative_path.as_posix(),
                        width=image.width,
                        height=image.height,
                        content_type=content_type,
                        sha256=hashlib.sha256(source_bytes).hexdigest(),
                        bytes=len(source_bytes),
                    )
                )
                if progress is not None:
                    progress.update(count, detail=f"cached: {entity.name}")
                continue
            try:
                source_bytes, content_type, final_url = _fetch_image_bytes(
                    config,
                    image,
                    client=client,
                    fetch_image=fetch_image,
                )
            except Exception:  # noqa: BLE001 - broken wiki media should not stop the pack build.
                if progress is not None:
                    progress.update(count, detail=f"skip image: {entity.name}")
                continue
            digest = hashlib.sha256(source_bytes).hexdigest()
            extension = _image_extension(final_url, content_type)
            relative_path = (
                Path("media")
                / "images"
                / entity.type
                / f"{_slug(entity.id)}__{index}{extension}"
            )
            output_path = config.output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(source_bytes)
            images.append(
                PackImage(
                    id=image_id,
                    entity_id=entity.id,
                    entity_type=entity.type,
                    title=entity.name,
                    alt=image.alt,
                    source_url=final_url,
                    page_url=page.url,
                    local_path=relative_path.as_posix(),
                    width=image.width,
                    height=image.height,
                    content_type=content_type,
                    sha256=digest,
                    bytes=len(source_bytes),
                )
            )
            if progress is not None:
                progress.update(count, detail=entity.name)
            if client is not None and config.delay_seconds > 0:
                time.sleep(config.delay_seconds)
    finally:
        if client is not None:
            client.close()
    if progress is not None:
        progress.finish(f"{len(images)} images")
    return images


def validate_hunt_pack_inputs(
    config: HuntWikiPackConfig,
    pages: Sequence[WikiPage],
    entities: Sequence[WikiEntity],
    chunks: Sequence[KnowledgeChunk],
    images: Sequence[PackImage],
) -> None:
    """Validate pack data before embeddings are computed.

    Embedding is the expensive step, so crawler/parser mistakes should fail here
    while the run can still be fixed without recomputing vectors.
    """

    errors: list[str] = []
    allowed_types = {value.lower() for value in config.allowed_entity_types}
    entity_by_id: dict[str, WikiEntity] = {}
    page_urls: set[str] = set()
    chunk_ids: set[str] = set()
    image_ids: set[str] = set()

    if len(pages) != len(entities):
        errors.append(f"page/entity count mismatch: {len(pages)} pages, {len(entities)} entities")
    if not entities:
        errors.append("no entities were produced")
    if not chunks:
        errors.append("no chunks were produced")

    for index, page in enumerate(pages):
        prefix = f"page[{index}]"
        if not page.title:
            errors.append(f"{prefix} has no title")
        if not page.page_name:
            errors.append(f"{prefix} has no page_name")
        if not page.url:
            errors.append(f"{prefix} has no url")
        elif page.url in page_urls:
            errors.append(f"{prefix} duplicates url {page.url}")
        page_urls.add(page.url)

    for index, entity in enumerate(entities):
        prefix = f"entity[{index}] {entity.id or '<missing-id>'}"
        if not entity.id:
            errors.append(f"{prefix} has no id")
        elif entity.id in entity_by_id:
            errors.append(f"{prefix} duplicates an entity id")
        else:
            entity_by_id[entity.id] = entity
        if not entity.type:
            errors.append(f"{prefix} has no type")
        elif entity.type not in allowed_types:
            errors.append(f"{prefix} has disallowed type {entity.type!r}")
        if entity.id and entity.type and not entity.id.startswith(f"{entity.type}:"):
            errors.append(f"{prefix} id does not start with its type")
        if not entity.name:
            errors.append(f"{prefix} has no name")
        if not entity.source_url:
            errors.append(f"{prefix} has no source_url")
        if not entity.page_name:
            errors.append(f"{prefix} has no page_name")
        if entity.source_url and page_urls and entity.source_url not in page_urls:
            errors.append(f"{prefix} source_url is not in pack pages")
        for alias in entity.aliases:
            if not _clean_text(alias):
                errors.append(f"{prefix} contains an empty alias")
        for skin_name in entity.skin_names:
            problem = _skin_name_validation_error(skin_name)
            if problem:
                errors.append(f"{prefix} has invalid skin name {skin_name!r}: {problem}")

    for index, chunk in enumerate(chunks):
        prefix = f"chunk[{index}] {chunk.id or '<missing-id>'}"
        if not chunk.id:
            errors.append(f"{prefix} has no id")
        elif chunk.id in chunk_ids:
            errors.append(f"{prefix} duplicates a chunk id")
        chunk_ids.add(chunk.id)
        entity = entity_by_id.get(chunk.entity_id)
        if entity is None:
            errors.append(f"{prefix} references unknown entity {chunk.entity_id!r}")
        elif chunk.entity_type != entity.type:
            errors.append(
                f"{prefix} entity_type {chunk.entity_type!r} does not match entity type {entity.type!r}"
            )
        if not _clean_text(chunk.text):
            errors.append(f"{prefix} has empty text")
        if not chunk.title:
            errors.append(f"{prefix} has no title")
        if not chunk.source_url:
            errors.append(f"{prefix} has no source_url")
        if chunk.source_url and page_urls and chunk.source_url not in page_urls:
            errors.append(f"{prefix} source_url is not in pack pages")

    entity_image_ids = {image_id for entity in entities for image_id in entity.image_ids}
    entity_image_paths = {path for entity in entities for path in entity.image_paths}
    for index, image in enumerate(images):
        prefix = f"image[{index}] {image.id or '<missing-id>'}"
        if not image.id:
            errors.append(f"{prefix} has no id")
        elif image.id in image_ids:
            errors.append(f"{prefix} duplicates an image id")
        image_ids.add(image.id)
        entity = entity_by_id.get(image.entity_id)
        if entity is None:
            errors.append(f"{prefix} references unknown entity {image.entity_id!r}")
        elif image.entity_type != entity.type:
            errors.append(
                f"{prefix} entity_type {image.entity_type!r} does not match entity type {entity.type!r}"
            )
        if not image.source_url:
            errors.append(f"{prefix} has no source_url")
        if not image.local_path:
            errors.append(f"{prefix} has no local_path")
        else:
            relative_path = Path(image.local_path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                errors.append(f"{prefix} has unsafe local_path {image.local_path!r}")
            elif not (config.output_dir / relative_path).exists():
                errors.append(f"{prefix} local_path does not exist: {image.local_path}")
        if image.bytes <= 0:
            errors.append(f"{prefix} has no bytes")
        if not image.sha256:
            errors.append(f"{prefix} has no sha256")
        if image.id and image.id not in entity_image_ids:
            errors.append(f"{prefix} is not attached to its entity image_ids")
        if image.local_path and image.local_path not in entity_image_paths:
            errors.append(f"{prefix} is not attached to its entity image_paths")

    missing_image_ids = entity_image_ids - image_ids
    if missing_image_ids:
        errors.append(f"entities reference missing image ids: {sorted(missing_image_ids)[:5]}")
    image_paths = {image.local_path for image in images}
    missing_image_paths = entity_image_paths - image_paths
    if missing_image_paths:
        errors.append(f"entities reference missing image paths: {sorted(missing_image_paths)[:5]}")

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:40])
        if len(errors) > 40:
            preview += f"\n- ... and {len(errors) - 40} more validation errors"
        raise RuntimeError(f"Hunt knowledge pack validation failed before embeddings:\n{preview}")


def _skin_name_validation_error(value: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return "empty"
    if cleaned[0].islower():
        return "starts with lowercase text"
    if ")" in cleaned:
        return "contains a dangling parenthesis"
    if ".." in cleaned:
        return "contains ellipsis text"
    if cleaned.isupper() and len(cleaned) > 1:
        return "looks like an uppercase section heading"
    if len(cleaned.split()) > 5:
        return "too many words"
    return ""


def parse_wiki_page(html: str, url: str, *, base_url: str = HUNT_WIKI_BASE_URL) -> WikiPage:
    parser = _WikiHTMLParser(base_url=base_url)
    parser.feed(html)
    parser.close()

    source_url = (
        normalize_wiki_article_url(parser.meta.get("canonical") or "", base_url)
        or normalize_wiki_article_url(url, base_url)
        or url
    )
    page_name = _page_name_from_url(source_url)
    title = _clean_text(parser.meta.get("og:title") or parser.first_heading or page_name.replace("_", " "))
    description = _clean_text(parser.meta.get("description") or parser.meta.get("og:description") or "")
    skin_names = _extract_skin_names_from_html(html)
    text_blocks = list(parser.blocks)
    if skin_names:
        text_blocks.append("## Skins " + ", ".join(skin_names))
    text = _clean_text("\n".join(text_blocks))
    links = [
        normalized
        for href in [*parser.links, *_extract_wiki_article_hrefs(html)]
        if (normalized := normalize_wiki_article_url(href, base_url)) is not None
    ]
    key_values = _key_values_from_rows(parser.druid_rows or parser.table_rows)
    if skin_names and "Skins" not in key_values:
        key_values["Skins"] = ", ".join(skin_names)
    return WikiPage(
        title=title,
        page_name=page_name,
        url=source_url,
        description=description,
        categories=_extract_categories(html),
        revision_id=_extract_revision_id(html),
        text=text,
        key_values=key_values,
        links=_dedupe(links),
        images=_dedupe_images(parser.images),
    )


def normalize_wiki_article_url(url: str, base_url: str = HUNT_WIKI_BASE_URL) -> str | None:
    if not url:
        return None
    url = unescape(url.strip())
    absolute = urljoin(base_url, url)
    absolute, _fragment = urldefrag(absolute)
    parsed = urlparse(absolute)
    base = urlparse(base_url)
    if parsed.netloc != base.netloc or not parsed.path.startswith("/wiki/"):
        return None
    if parsed.query:
        return None
    page_name = unquote(parsed.path.removeprefix("/wiki/"))
    if not page_name:
        return None
    namespace = page_name.split(":", 1)[0].lower() if ":" in page_name else ""
    if namespace in {
        "file",
        "special",
        "talk",
        "user",
        "template",
        "module",
        "mediawiki",
        "data",
        "help",
        "meta",
    }:
        return None
    return f"{base.scheme}://{base.netloc}/wiki/{page_name}"


def _extract_wiki_article_hrefs(html: str) -> list[str]:
    return [
        unescape(match.group(1))
        for match in re.finditer(r"""<a\b[^>]*\bhref=["']([^"']*/wiki/[^"'#?]+)["']""", html or "", flags=re.IGNORECASE)
    ]


def _extract_skin_names_from_html(html: str) -> list[str]:
    section = _extract_skins_html_section(html)
    if not section:
        return []
    values: list[str] = []
    for match in re.finditer(
        r"""<a\b[^>]*\bclass=["'][^"']*\btabber__tab\b[^"']*["'][^>]*>(.*?)</a>""",
        section,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        values.append(_html_text(match.group(1)))
    for match in re.finditer(
        r"""<div\b[^>]*\bclass=["'][^"']*\bdruid-title\b[^"']*["'][^>]*>(.*?)</div>""",
        section,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        values.append(_html_text(match.group(1)))
    return _dedupe(value for value in values if value and value.lower() != "base weapon")


def _extract_skins_html_section(html: str) -> str:
    match = re.search(
        r"""<span\b[^>]*\bid=["']Skins["'][^>]*>.*?</span>""",
        html or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    start = match.end()
    tail = html[start:]
    end_match = re.search(
        r"""<h2\b[^>]*>\s*<span\b[^>]*\bid=["'](?:Equipment_Animations|Book_of_Weapons|Update_History|Gallery|Lore_Connections)["']""",
        tail,
        flags=re.IGNORECASE | re.DOTALL,
    )
    end = end_match.start() if end_match else len(tail)
    return tail[:end]


def _html_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value or "")
    return _clean_text(without_tags)


class _WikiClient:
    def __init__(self, config: HuntWikiPackConfig) -> None:
        self.config = config
        self._client = httpx.Client(
            follow_redirects=True,
            timeout=config.timeout_seconds,
            headers={"User-Agent": config.user_agent},
        )
        self._robots = robotparser.RobotFileParser()
        response = self._client.get(urljoin(config.base_url, "/robots.txt"))
        response.raise_for_status()
        self._robots.parse(response.text.splitlines())

    def can_fetch(self, url: str) -> bool:
        return self._robots.can_fetch(self.config.user_agent, url)

    def fetch(self, url: str) -> str:
        response = self._client.get(url)
        response.raise_for_status()
        if "text/html" not in response.headers.get("content-type", ""):
            raise RuntimeError(f"Expected HTML response for {url}")
        return response.text

    def fetch_bytes(self, url: str, *, max_bytes: int) -> tuple[bytes, str, str]:
        response = self._client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if not content_type.startswith("image/"):
            raise RuntimeError(f"Expected image response for {url}, got {content_type or 'unknown'}")
        content = response.content
        if len(content) > max_bytes:
            raise RuntimeError(f"Image response exceeds {max_bytes} bytes: {url}")
        return content, content_type, str(response.url)

    def close(self) -> None:
        self._client.close()


class _BrowserPageFetcher:
    def __init__(self, config: HuntWikiPackConfig) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise RuntimeError("Playwright is required for --browser-fetch.") from exc

        self.config = config
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=not config.show_browser_on_block,
            args=["--disable-blink-features=AutomationControlled"],
        )
        self._context = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 1000},
            locale="en-US",
            timezone_id="Europe/Berlin",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        self._context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        self._page = self._context.new_page()
        self._last_load_at = 0.0

    def fetch(self, url: str) -> str:
        html, status, content_type = self._load(url)
        if self._needs_manual_unblock(status, html) and self.config.show_browser_on_block:
            html, status, content_type = self._wait_for_manual_unblock(url)
        if self._is_error_response(status, html):
            request = httpx.Request("GET", url)
            response_obj = httpx.Response(
                status or 403,
                request=request,
                headers={"content-type": content_type or "text/html; charset=UTF-8"},
                text=html,
            )
            raise httpx.HTTPStatusError(
                f"Browser fetch returned HTTP {status or 403} for {url}",
                request=request,
                response=response_obj,
            )
        if "text/html" not in content_type:
            raise RuntimeError(f"Expected HTML response for {url}")
        if not self._looks_like_article(html):
            raise RuntimeError(f"Browser fetch did not produce wiki article HTML for {url}")
        return html

    def _load(self, url: str) -> tuple[str, int, str]:
        self._pause_before_load()
        response = self._page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=max(self.config.timeout_seconds, 1.0) * 1000,
        )
        self._page.wait_for_load_state("domcontentloaded", timeout=max(self.config.timeout_seconds, 1.0) * 1000)
        html = self._page.content()
        status = response.status if response is not None else 0
        content_type = response.headers.get("content-type", "") if response is not None else "text/html"
        return html, status, content_type

    def _wait_for_manual_unblock(self, url: str) -> tuple[str, int, str]:
        self._page.bring_to_front()
        sys.stderr.write(
            "\nBrowser fetch was blocked. Complete the wiki.gg check in the visible Chromium window; "
            f"the crawler will continue when the article page loads. If the page stays blocked after "
            f"the check, refresh it.\nBlocked URL: {url}\n"
        )
        sys.stderr.flush()
        content_type = "text/html"
        deadline = time.monotonic() + 15 * 60
        while time.monotonic() < deadline:
            self._page.wait_for_timeout(1000)
            try:
                html = self._page.content()
            except Exception:  # noqa: BLE001 - the challenge page can navigate while we poll.
                continue
            if not self._looks_like_article(html):
                continue
            current_url = normalize_wiki_article_url(self._page.url, self.config.base_url)
            target_url = normalize_wiki_article_url(url, self.config.base_url)
            if current_url == target_url:
                return html, 200, content_type
            try:
                html, status, content_type = self._load(url)
            except Exception:  # noqa: BLE001 - keep waiting while the user is solving the check.
                continue
            if self._looks_like_article(html) and not self._needs_manual_unblock(status, html):
                return html, status, content_type
        try:
            return self._page.content(), 403, content_type
        except Exception:  # noqa: BLE001 - preserve the block error path if the page is mid-navigation.
            return "", 403, content_type

    def _pause_before_load(self) -> None:
        delay = max(float(self.config.delay_seconds), 0.0)
        if delay <= 0.0:
            return
        elapsed = time.monotonic() - self._last_load_at
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_load_at = time.monotonic()

    @staticmethod
    def _is_error_response(status: int, html: str) -> bool:
        return status >= 400 or _BrowserPageFetcher._has_block_marker(html)

    @staticmethod
    def _needs_manual_unblock(status: int, html: str) -> bool:
        return status in {403, 429} or _BrowserPageFetcher._has_block_marker(html)

    @staticmethod
    def _has_block_marker(html: str) -> bool:
        lowered = html.lower()
        return any(
            marker in lowered
            for marker in (
                "blocked - wiki.gg",
                "wiki.gg has blocked",
                "checking if the site connection is secure",
                "checking your browser",
                "attention required",
            )
        )

    @staticmethod
    def _looks_like_article(html: str) -> bool:
        return (
            "mw-parser-output" in html
            or 'id="mw-content-text"' in html
            or "id='mw-content-text'" in html
        ) and not _BrowserPageFetcher._has_block_marker(html)

    def close(self) -> None:
        self._context.close()
        self._browser.close()
        self._playwright.stop()


class _SeleniumPageFetcher:
    def __init__(self, config: HuntWikiPackConfig) -> None:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
        except ModuleNotFoundError as exc:
            raise RuntimeError("Selenium is required for --selenium-fetch.") from exc

        self.config = config
        self._last_load_at = 0.0
        custom_factory = os.getenv("HUNT_WIKI_SELENIUM_DRIVER_FACTORY", "").strip()
        if custom_factory:
            self._driver = self._load_driver_factory(custom_factory)(config)
            return

        options = Options()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1440,1000")
        options.add_argument(f"--user-agent={config.user_agent}")
        if config.selenium_headless and not config.show_browser_on_block:
            options.add_argument("--headless=new")
        if config.selenium_profile_dir is not None:
            config.selenium_profile_dir.mkdir(parents=True, exist_ok=True)
            options.add_argument(f"--user-data-dir={config.selenium_profile_dir}")
        if config.selenium_remote_url:
            self._driver = webdriver.Remote(command_executor=config.selenium_remote_url, options=options)
        else:
            self._driver = webdriver.Chrome(options=options)

    def fetch(self, url: str) -> str:
        html, status, content_type = self._load(url)
        if self._needs_manual_unblock(status, html) and self.config.show_browser_on_block:
            html, status, content_type = self._wait_for_manual_unblock(url)
        if self._is_error_response(status, html):
            request = httpx.Request("GET", url)
            response_obj = httpx.Response(
                status or 403,
                request=request,
                headers={"content-type": content_type or "text/html; charset=UTF-8"},
                text=html,
            )
            raise httpx.HTTPStatusError(
                f"Selenium fetch returned HTTP {status or 403} for {url}",
                request=request,
                response=response_obj,
            )
        if "text/html" not in content_type:
            raise RuntimeError(f"Expected HTML response for {url}")
        if not self._looks_like_article(html):
            raise RuntimeError(f"Selenium fetch did not produce wiki article HTML for {url}")
        return html

    def _load(self, url: str) -> tuple[str, int, str]:
        self._pause_before_load()
        self._driver.get(url)
        self._wait_for_ready_state()
        return self._driver.page_source, self._status_code(), "text/html"

    def _wait_for_manual_unblock(self, url: str) -> tuple[str, int, str]:
        sys.stderr.write(
            "\nSelenium fetch was blocked. Complete the wiki.gg check in the browser window; "
            "the crawler will continue when the article page loads. If the page stays blocked, "
            f"refresh it.\nBlocked URL: {url}\n"
        )
        sys.stderr.flush()
        deadline = time.monotonic() + 15 * 60
        target_url = normalize_wiki_article_url(url, self.config.base_url)
        while time.monotonic() < deadline:
            time.sleep(1.0)
            html = self._driver.page_source
            if not self._looks_like_article(html):
                continue
            current_url = normalize_wiki_article_url(str(self._driver.current_url), self.config.base_url)
            if current_url == target_url:
                return html, 200, "text/html"
            try:
                html, status, content_type = self._load(url)
            except Exception:  # noqa: BLE001 - keep waiting while the user is solving the check.
                continue
            if self._looks_like_article(html) and not self._needs_manual_unblock(status, html):
                return html, status, content_type
        return self._driver.page_source, 403, "text/html"

    def _wait_for_ready_state(self) -> None:
        deadline = time.monotonic() + max(self.config.timeout_seconds, 1.0)
        while time.monotonic() < deadline:
            try:
                state = self._driver.execute_script("return document.readyState")
            except Exception:  # noqa: BLE001 - page may still be initializing.
                time.sleep(0.2)
                continue
            if state in {"interactive", "complete"}:
                return
            time.sleep(0.2)

    def _status_code(self) -> int:
        try:
            entries = self._driver.execute_script(
                "return performance.getEntriesByType('navigation').map(e => e.responseStatus || 0)"
            )
        except Exception:  # noqa: BLE001 - Selenium drivers do not always expose response status.
            return 200
        if isinstance(entries, list) and entries:
            try:
                return int(entries[-1]) or 200
            except (TypeError, ValueError):
                return 200
        return 200

    def _pause_before_load(self) -> None:
        delay = max(float(self.config.delay_seconds), 0.0)
        if delay <= 0.0:
            return
        elapsed = time.monotonic() - self._last_load_at
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_load_at = time.monotonic()

    @staticmethod
    def _load_driver_factory(spec: str) -> Callable[[HuntWikiPackConfig], object]:
        module_name, separator, function_name = spec.partition(":")
        if not separator or not module_name or not function_name:
            raise RuntimeError("HUNT_WIKI_SELENIUM_DRIVER_FACTORY must be formatted as module:function.")
        module = importlib.import_module(module_name)
        factory = getattr(module, function_name, None)
        if not callable(factory):
            raise RuntimeError(f"Selenium driver factory is not callable: {spec}")
        return factory

    @staticmethod
    def _is_error_response(status: int, html: str) -> bool:
        return status >= 400 or _SeleniumPageFetcher._has_block_marker(html)

    @staticmethod
    def _needs_manual_unblock(status: int, html: str) -> bool:
        return status in {403, 429} or _SeleniumPageFetcher._has_block_marker(html)

    @staticmethod
    def _has_block_marker(html: str) -> bool:
        return _BrowserPageFetcher._has_block_marker(html)

    @staticmethod
    def _looks_like_article(html: str) -> bool:
        return _BrowserPageFetcher._looks_like_article(html)

    def close(self) -> None:
        self._driver.quit()


class _WikiHTMLParser(HTMLParser):
    def __init__(self, *, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.meta: dict[str, str] = {}
        self.first_heading = ""
        self.blocks: list[str] = []
        self.links: list[str] = []
        self.images: list[WikiImage] = []
        self.table_rows: list[list[str]] = []
        self.druid_rows: list[list[str]] = []
        self._in_title = False
        self._title_parts: list[str] = []
        self._in_first_heading = False
        self._first_heading_parts: list[str] = []
        self._in_content = False
        self._content_depth = 0
        self._ignored_depths: list[int] = []
        self._block_parts: list[str] = []
        self._heading_level: int | None = None
        self._heading_parts: list[str] = []
        self._row_cells: list[str] | None = None
        self._cell_parts: list[str] | None = None
        self._druid_label_parts: list[str] | None = None
        self._druid_label_depth: int | None = None
        self._druid_data_parts: list[str] | None = None
        self._druid_data_depth: int | None = None
        self._pending_druid_label: str | None = None

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {key.lower(): value or "" for key, value in attrs_list}
        tag = tag.lower()
        if tag == "meta":
            self._capture_meta(attrs)
            return
        if tag == "link" and attrs.get("rel") == "canonical" and attrs.get("href"):
            self.meta["canonical"] = urljoin(self.base_url, attrs["href"])
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "h1" and attrs.get("id") == "firstHeading":
            self._in_first_heading = True
            return
        if not self._in_content and tag == "div" and "mw-parser-output" in attrs.get("class", ""):
            self._in_content = True
            self._content_depth = 1
            return
        if not self._in_content:
            return

        self._content_depth += 1
        class_name = attrs.get("class", "")
        if tag in {"script", "style"} or _is_ignored_content_class(class_name):
            self._ignored_depths.append(self._content_depth)
            return
        if self._ignored_depths:
            return
        if tag == "a" and attrs.get("href"):
            self.links.append(urljoin(self.base_url, attrs["href"]))
        if tag == "img" and attrs.get("src"):
            self._capture_image(attrs)
        if tag in {"p", "div", "ul", "ol", "li", "table", "tr", "h2", "h3", "h4", "h5", "h6"}:
            self._flush_block()
        if tag in {"h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = int(tag[1])
            self._heading_parts = []
        if tag == "div" and "druid-label" in class_name:
            self._druid_label_parts = []
            self._druid_label_depth = self._content_depth
        if tag == "div" and "druid-data" in class_name:
            self._druid_data_parts = []
            self._druid_data_depth = self._content_depth
        if tag == "tr":
            self._row_cells = []
        if tag in {"td", "th"} and self._row_cells is not None:
            self._cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            if self._title_parts:
                self.meta.setdefault("title", _clean_text("".join(self._title_parts)))
            return
        if tag == "h1" and self._in_first_heading:
            self._in_first_heading = False
            self.first_heading = _clean_text("".join(self._first_heading_parts))
            return
        if not self._in_content:
            return

        if self._ignored_depths and self._ignored_depths[-1] == self._content_depth:
            self._ignored_depths.pop()
        elif not self._ignored_depths:
            if tag == "div" and self._druid_label_depth == self._content_depth and self._druid_label_parts is not None:
                label = _clean_text("".join(self._druid_label_parts)).rstrip(":")
                if label:
                    self._pending_druid_label = label
                self._druid_label_parts = None
                self._druid_label_depth = None
            elif tag == "div" and self._druid_data_depth == self._content_depth and self._druid_data_parts is not None:
                value = _clean_text("".join(self._druid_data_parts))
                if self._pending_druid_label and value:
                    self.druid_rows.append([self._pending_druid_label, value])
                self._pending_druid_label = None
                self._druid_data_parts = None
                self._druid_data_depth = None
            elif tag in {"td", "th"} and self._cell_parts is not None and self._row_cells is not None:
                cell = _clean_text("".join(self._cell_parts))
                if cell:
                    self._row_cells.append(cell)
                self._cell_parts = None
            elif tag == "tr" and self._row_cells is not None:
                if self._row_cells:
                    self.table_rows.append(self._row_cells)
                self._row_cells = None
            elif self._heading_level is not None and tag == f"h{self._heading_level}":
                heading = _clean_text("".join(self._heading_parts))
                if heading:
                    self.blocks.append(f"{'#' * self._heading_level} {heading}")
                self._heading_level = None
                self._heading_parts = []
            elif tag in {"p", "div", "li", "table", "ul", "ol"}:
                self._flush_block()

        self._content_depth -= 1
        if self._content_depth <= 0:
            self._flush_block()
            self._in_content = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        if self._in_first_heading:
            self._first_heading_parts.append(data)
        if not self._in_content or self._ignored_depths:
            return
        if self._druid_label_parts is not None:
            self._druid_label_parts.append(data)
        if self._druid_data_parts is not None:
            self._druid_data_parts.append(data)
        if self._cell_parts is not None:
            self._cell_parts.append(data)
        if self._heading_level is not None:
            self._heading_parts.append(data)
            return
        self._block_parts.append(data)

    def _capture_meta(self, attrs: dict[str, str]) -> None:
        key = attrs.get("name") or attrs.get("property")
        content = attrs.get("content")
        if key and content and key in {"description", "og:title", "og:description", "og:url"}:
            self.meta[key] = unescape(content)

    def _flush_block(self) -> None:
        text = _clean_text("".join(self._block_parts))
        self._block_parts = []
        if len(text) >= 3:
            self.blocks.append(text)

    def _capture_image(self, attrs: dict[str, str]) -> None:
        width = _parse_optional_int(attrs.get("data-file-width") or attrs.get("width"))
        height = _parse_optional_int(attrs.get("data-file-height") or attrs.get("height"))
        self.images.append(
            WikiImage(
                url=urljoin(self.base_url, attrs["src"]),
                alt=_clean_text(attrs.get("alt") or attrs.get("title") or ""),
                width=width,
                height=height,
            )
        )


def _create_pack_embedder(config: HuntWikiPackConfig) -> HuggingFaceMultimodalEmbedder:
    settings = get_settings()
    backend = _real_embedding_backend(settings) if config.require_real_embeddings else None
    return HuggingFaceMultimodalEmbedder(
        EmbeddingConfig(
            model_name=settings.tier.multimodal_retrieval_model,
            dimension=config.embedding_dimension,
            modality="text",
            mock_fallback=not config.require_real_embeddings,
            model_path=_embedding_model_path(settings),
            runtime_backend=_runtime_backend(settings),
            precision=_embedding_precision(settings),
        ),
        backend=backend,
    )


def _real_embedding_backend(settings: AppSettings) -> TransformersEmbeddingBackend:
    return TransformersEmbeddingBackend(
        model_for_role("embedder", settings.model_tier, device_backend=settings.gpu_backend),
        manager=transformers_runtime_manager(
            models_dir=settings.models_dir,
            logs_dir=settings.logs_dir,
            gpu_backend=settings.gpu_backend,
            one_model_at_a_time=True,
            torch_compile_mode=settings.torch_compile_mode,
            torch_compile_backend=settings.torch_compile_backend,
            torch_compile_profile=settings.torch_compile_profile,
            generation_cache_implementation=settings.qwen_cache_implementation,
        ),
        batch_size=16,
    )


def _embed_chunks(
    embedder: HuggingFaceMultimodalEmbedder,
    chunks: Sequence[KnowledgeChunk],
    *,
    batch_size: int,
    progress: _ProgressReporter | None = None,
) -> np.ndarray:
    vectors: list[list[float]] = []
    batch = max(batch_size, 1)
    for start in range(0, len(chunks), batch):
        texts = [chunk.text for chunk in chunks[start : start + batch]]
        vectors.extend(embedder.embed_texts(texts))
        if progress is not None:
            progress.update(min(start + batch, len(chunks)), detail=f"batch {start // batch + 1}")
    return np.asarray(vectors, dtype=np.float32)


def _write_pack(
    output_dir: Path,
    config: HuntWikiPackConfig,
    pages: Sequence[WikiPage],
    entities: Sequence[WikiEntity],
    chunks: Sequence[KnowledgeChunk],
    images: Sequence[PackImage],
    vectors: np.ndarray,
) -> None:
    _write_jsonl(output_dir / "entities.jsonl", (asdict(entity) for entity in entities))
    _write_jsonl(output_dir / "chunks.jsonl", (asdict(chunk) for chunk in chunks))
    _write_jsonl(output_dir / "media_index.jsonl", (asdict(image) for image in images))
    _write_jsonl(
        output_dir / "embedding_index.jsonl",
        (
            {
                "offset": index,
                "chunk_id": chunk.id,
                "entity_id": chunk.entity_id,
                "entity_type": chunk.entity_type,
                "source_url": chunk.source_url,
            }
            for index, chunk in enumerate(chunks)
        ),
    )
    _write_jsonl(
        output_dir / "attribution.jsonl",
        (
            {
                "title": page.title,
                "page_name": page.page_name,
                "source_url": page.url,
                "revision_id": page.revision_id,
                "license": HUNT_WIKI_LICENSE,
                "license_url": HUNT_WIKI_LICENSE_URL,
            }
            for page in pages
        ),
    )
    np.save(output_dir / "embeddings.npy", vectors)
    (output_dir / "manifest.json").write_text(
        json.dumps(_manifest(config, pages, entities, chunks, images, vectors), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _manifest(
    config: HuntWikiPackConfig,
    pages: Sequence[WikiPage],
    entities: Sequence[WikiEntity],
    chunks: Sequence[KnowledgeChunk],
    images: Sequence[PackImage],
    vectors: np.ndarray,
) -> dict[str, object]:
    settings = get_settings()
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "name": "huntshowdown-wiki-knowledge-pack",
        "source": config.base_url,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "entity_count": len(entities),
        "chunk_count": len(chunks),
        "image_count": len(images),
        "embedding_model": settings.tier.multimodal_retrieval_model,
        "embedding_profile": HF_RETRIEVAL_PROFILE,
        "embedding_modality": "text",
        "embedding_dimension": int(vectors.shape[1]) if vectors.ndim == 2 else config.embedding_dimension,
        "embedding_file": "embeddings.npy",
        "embedding_index_file": "embedding_index.jsonl",
        "entities_file": "entities.jsonl",
        "chunks_file": "chunks.jsonl",
        "crawl_cache_file": "crawl_pages.jsonl",
        "media_index_file": "media_index.jsonl",
        "media_dir": "media/images",
        "attribution_file": "attribution.jsonl",
        "license": HUNT_WIKI_LICENSE,
        "license_url": HUNT_WIKI_LICENSE_URL,
        "content_notes": [
            "Derived from normal Hunt wiki article pages, not api.php.",
            "Text is stored for local search/RAG use with source attribution.",
            "Images are stored from article HTML image URLs."
            if config.include_images
            else "No wiki images are included in this text-only pack.",
        ],
        "crawl": {
            "seeds": list(config.seeds),
            "allowed_entity_types": list(config.allowed_entity_types),
            "max_pages": config.max_pages,
            "max_depth": config.max_depth,
            "delay_seconds": config.delay_seconds,
            "crawl_concurrency": config.crawl_concurrency,
            "include_images": config.include_images,
            "max_images_per_page": config.max_images_per_page,
            "reuse_crawl_cache": config.reuse_crawl_cache,
            "browser_fetch": config.browser_fetch,
            "selenium_fetch": config.selenium_fetch,
            "selenium_remote_url": config.selenium_remote_url,
            "selenium_headless": config.selenium_headless,
            "selenium_profile_dir": str(config.selenium_profile_dir) if config.selenium_profile_dir else "",
            "show_browser_on_block": config.show_browser_on_block,
            "user_agent": config.user_agent,
        },
        "sha256": {
            "entities.jsonl": _file_sha256(config.output_dir / "entities.jsonl"),
            "chunks.jsonl": _file_sha256(config.output_dir / "chunks.jsonl"),
            "embedding_index.jsonl": _file_sha256(config.output_dir / "embedding_index.jsonl"),
            "media_index.jsonl": _file_sha256(config.output_dir / "media_index.jsonl"),
            "embeddings.npy": _file_sha256(config.output_dir / "embeddings.npy"),
            "attribution.jsonl": _file_sha256(config.output_dir / "attribution.jsonl"),
            "crawl_pages.jsonl": _file_sha256(config.output_dir / "crawl_pages.jsonl"),
        }
        if (config.output_dir / "embeddings.npy").exists()
        else {},
    }


def _entity_from_page(page: WikiPage) -> WikiEntity:
    entity_type = _infer_entity_type(page)
    name = _display_name(page)
    skin_names = _skin_names_for_page(page, name, entity_type)
    aliases = _dedupe([*_aliases_for_page(page, name), *skin_names])
    key_values = {
        key: value
        for key, value in page.key_values.items()
        if key.lower() not in {"image", "icon", "source"}
    }
    if skin_names and "Skins" not in key_values:
        key_values["Skins"] = ", ".join(skin_names)
    return WikiEntity(
        id=f"{entity_type}:{_slug(page.page_name)}",
        type=entity_type,
        name=name,
        aliases=aliases,
        description=page.description,
        source_url=page.url,
        page_name=page.page_name,
        categories=page.categories,
        key_values=key_values,
        skin_names=skin_names,
    )


def _attach_images_to_entities(
    entities: Sequence[WikiEntity],
    images: Sequence[PackImage],
) -> list[WikiEntity]:
    by_entity: dict[str, list[PackImage]] = {}
    for image in images:
        by_entity.setdefault(image.entity_id, []).append(image)
    output: list[WikiEntity] = []
    for entity in entities:
        entity_images = by_entity.get(entity.id, [])
        output.append(
            replace(
                entity,
                image_ids=[image.id for image in entity_images],
                image_paths=[image.local_path for image in entity_images],
            )
        )
    return output


def _select_page_images(
    config: HuntWikiPackConfig,
    page: WikiPage,
    entity: WikiEntity,
) -> list[WikiImage]:
    candidates = [
        image
        for image in page.images
        if _is_relevant_image(config, image, entity)
    ]
    ranked = sorted(candidates, key=lambda image: _image_score(image, entity), reverse=True)
    return ranked[: max(config.max_images_per_page, 0)]


def _is_relevant_image(
    config: HuntWikiPackConfig,
    image: WikiImage,
    entity: WikiEntity,
) -> bool:
    parsed = urlparse(image.url)
    if parsed.netloc and parsed.netloc != urlparse(config.base_url).netloc:
        return False
    if "/images/" not in parsed.path:
        return False
    filename = _image_filename(image.url).lower()
    if not filename:
        return False
    excluded_prefixes = (
        "site-",
        "currency_",
        "icon_filter_",
        "scarce_icon",
        "stub.",
        "network_",
    )
    if filename.startswith(excluded_prefixes):
        return False
    long_side = max(image.width or 0, image.height or 0)
    if long_side and long_side < config.min_image_long_side:
        return False
    if entity.type in {"weapon", "tool", "consumable"}:
        cleaned = _clean_skin_source_name(image.alt or filename, entity.type)
        cleaned_key = _slug(cleaned)
        entity_key = _slug(entity.name)
        if cleaned_key == entity_key:
            return True
        return _is_skin_like_image(image) and cleaned_key.startswith(entity_key + "-")
    if entity.type == "map":
        return True
    included_prefixes = (
        "weapon_",
        "tool_",
        "consumable_",
        "ammo_",
        "trait_",
        "map_",
        "target_",
        "monster_",
        "world_item_",
    )
    return filename.startswith(included_prefixes)


def _is_skin_like_image(image: WikiImage) -> bool:
    haystack = f"{image.alt} {_image_filename(image.url)} {image.url}".lower()
    return any(marker in haystack for marker in (" 3d ", "_3d_", "model", "dlc", "promo"))


def _image_score(image: WikiImage, entity: WikiEntity) -> tuple[int, int]:
    filename = _image_filename(image.url).lower()
    entity_slug = _slug(entity.name).replace("-", "_")
    score = 0
    if "/thumb/" not in urlparse(image.url).path:
        score += 80
    if entity_slug and entity_slug in filename:
        score += 60
    if filename.startswith(f"{entity.type}_"):
        score += 40
    if image.alt and entity.name.lower() in image.alt.lower():
        score += 25
    return score, max(image.width or 0, image.height or 0)


def _fetch_image_bytes(
    config: HuntWikiPackConfig,
    image: WikiImage,
    *,
    client: _WikiClient | None,
    fetch_image: FetchBytes | None,
) -> tuple[bytes, str, str]:
    last_error: Exception | None = None
    for url in _candidate_image_urls(image.url):
        try:
            if client is not None:
                if not client.can_fetch(url):
                    continue
                return client.fetch_bytes(url, max_bytes=config.max_image_bytes)
            if fetch_image is None:
                raise RuntimeError("No image fetcher is available.")
            content = fetch_image(url)
            if len(content) > config.max_image_bytes:
                raise RuntimeError(f"Image response exceeds {config.max_image_bytes} bytes: {url}")
            content_type = mimetypes.guess_type(urlparse(url).path)[0] or "application/octet-stream"
            return content, content_type, url
        except Exception as exc:  # noqa: BLE001 - try thumbnail fallback when original fails.
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No allowed image URL for {image.url}")


def _existing_image_file(output_dir: Path, entity: WikiEntity, index: int) -> Path | None:
    image_dir = output_dir / "media" / "images" / entity.type
    prefix = f"{_slug(entity.id)}__{index}"
    matches = sorted(path for path in image_dir.glob(f"{prefix}.*") if path.is_file())
    return matches[0] if matches else None


def _is_skippable_page_error(exc: httpx.HTTPStatusError) -> bool:
    status = exc.response.status_code
    return status in {404, 410}


def _candidate_image_urls(url: str) -> list[str]:
    parsed = urlparse(url)
    path = parsed.path
    candidates = []
    if "/images/thumb/" in path:
        thumb_tail = path.split("/images/thumb/", 1)[1]
        original_name = thumb_tail.rsplit("/", 1)[0]
        original_url = parsed._replace(path=f"/images/{original_name}").geturl()
        candidates.append(original_url)
    candidates.append(url)
    return _dedupe(candidates)


def _image_filename(url: str) -> str:
    path = unquote(urlparse(url).path)
    if "/images/thumb/" in path:
        tail = path.split("/images/thumb/", 1)[1]
        return tail.rsplit("/", 1)[0].rsplit("/", 1)[-1]
    return path.rsplit("/", 1)[-1]


def _image_extension(url: str, content_type: str) -> str:
    path_extension = Path(urlparse(url).path).suffix.lower()
    if path_extension in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
        return path_extension
    guessed = mimetypes.guess_extension(content_type)
    if guessed:
        return ".jpg" if guessed == ".jpe" else guessed
    return ".bin"


def _chunks_for_page(
    page: WikiPage,
    entity: WikiEntity,
    *,
    max_words: int = 220,
    overlap: int = 35,
) -> list[KnowledgeChunk]:
    words = _knowledge_text(page, entity).split()
    if not words:
        return []
    chunks: list[KnowledgeChunk] = []
    start = 0
    index = 0
    step = max(max_words - overlap, 1)
    while start < len(words):
        selected = words[start : start + max_words]
        chunks.append(
            KnowledgeChunk(
                id=f"{entity.id}:chunk:{index}",
                entity_id=entity.id,
                entity_type=entity.type,
                title=entity.name,
                text=" ".join(selected),
                source_url=page.url,
                page_name=page.page_name,
                revision_id=page.revision_id,
            )
        )
        if start + max_words >= len(words):
            break
        start += step
        index += 1
    return chunks


def _knowledge_text(page: WikiPage, entity: WikiEntity) -> str:
    key_values = "\n".join(
        f"{key}: {value}" for key, value in list(entity.key_values.items())[:24] if value
    )
    parts = [
        f"Name: {entity.name}",
        f"Type: {entity.type}",
        f"Aliases: {', '.join(entity.aliases)}" if entity.aliases else "",
        f"Categories: {', '.join(page.categories)}" if page.categories else "",
        f"Description: {page.description}" if page.description else "",
        key_values,
        page.text,
    ]
    return _clean_text("\n".join(part for part in parts if part))


def _infer_entity_type(page: WikiPage) -> str:
    page_name = page.page_name.lower()
    categories = {category.lower() for category in page.categories}
    if page_name.startswith("weapons/") or "weapons" in categories or "book of weapons" in categories:
        return "weapon"
    if page_name.startswith("maps/"):
        return "map"
    if page_name.startswith("traits/") or "traits" in categories:
        return "trait"
    if page_name.startswith("tools/") or "tools" in categories:
        return "tool"
    if page_name.startswith("consumables/") or "consumables" in categories:
        return "consumable"
    if page_name.startswith("targets/") or "targets" in categories:
        return "target"
    if page_name.startswith("monsters/") or "monsters" in categories:
        return "monster"
    if page_name.startswith("world_items/") or "world items" in categories:
        return "world_item"
    if "ammo" in page_name or any("ammo" in category for category in categories):
        return "ammo"
    if page_name.startswith("category:"):
        return "category"
    return "article"


def _display_name(page: WikiPage) -> str:
    title = page.title.replace(" - Hunt: Showdown 1896 Wiki", "").strip()
    if title and not title.startswith("Category:"):
        if "/" in title:
            root, tail = title.split("/", 1)
            if root.lower() in {"weapons", "tools", "consumables", "maps"}:
                return tail.replace("/", " ")
        return title
    page_name = page.page_name.replace("_", " ")
    if "/" in page_name:
        root, tail = page_name.split("/", 1)
        if root.lower() in {"weapons", "tools", "consumables", "maps"}:
            return tail.replace("/", " ")
        return page_name.rsplit("/", 1)[-1]
    if title:
        return title
    return page_name


def _aliases_for_page(page: WikiPage, name: str) -> list[str]:
    values = [
        name,
        page.title,
        page.page_name.replace("_", " "),
        page.page_name.rsplit("/", 1)[-1].replace("_", " "),
    ]
    tokens = name.split()
    if len(tokens) > 1:
        values.append(tokens[0])
        values.append(tokens[-1])
    return _dedupe(_clean_text(value) for value in values if value)


def _skin_names_for_page(page: WikiPage, entity_name: str, entity_type: str) -> list[str]:
    if entity_type not in {"weapon", "tool", "consumable"}:
        return []
    values: list[str] = []
    for image in page.images:
        values.extend(_skin_names_from_image(image, entity_name, entity_type))
    values.extend(_skin_names_from_text_section(page.text, entity_name, entity_type))
    values.extend(_skin_names_from_text_section(page.description, entity_name, entity_type))
    if skins := page.key_values.get("Skins"):
        values.extend(_skin_names_from_compact_list(skins, entity_name, entity_type))
    return _dedupe(_normalize_skin_name(value, entity_name) for value in values)


def _skin_names_from_image(image: WikiImage, entity_name: str, entity_type: str) -> list[str]:
    raw_values = [image.alt, _image_filename(image.url)]
    output: list[str] = []
    for raw in raw_values:
        if not raw:
            continue
        raw_lower = raw.lower()
        if not any(marker in raw_lower for marker in (" 3d ", "_3d_", "model", "dlc", "promo")):
            continue
        cleaned = _clean_skin_source_name(raw, entity_type)
        normalized = _normalize_skin_name(cleaned, entity_name)
        if normalized:
            output.append(normalized)
    return output


def _skin_names_from_text_section(text: str, entity_name: str, entity_type: str) -> list[str]:
    value = text or ""
    match = re.search(r"##\s*Skins\s+(.*?)(?:\s+##\s+|$)", value, flags=re.IGNORECASE)
    if match:
        return _skin_names_from_compact_list(match.group(1), entity_name, entity_type)
    marker = rf"\bBase\s+{re.escape(entity_type)}"
    match = re.search(marker + r"(.+)$", value, flags=re.IGNORECASE)
    if match:
        return _skin_names_from_compact_list(match.group(1), entity_name, entity_type)
    return []


def _skin_names_from_compact_list(value: str, entity_name: str, entity_type: str) -> list[str]:
    cleaned = re.sub(rf"\bBase\s+{re.escape(entity_type)}", " ", value, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bBase\s+Weapon|\bBase\s+Tool|\bBase\s+Consumable", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.split(
        r"\b(?:Promo|DLC Art|Model|Price|Source|Update|Trivia|Book of Weapons|Report bad advertisement|Cookies help)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    cleaned = re.sub(r"(?<=[a-z’'])(?=[A-Z])", "|", cleaned)
    cleaned = re.sub(r"(?<=[a-z])(?=\d)", "|", cleaned)
    pieces = re.split(r"\s{2,}|[,;|]", cleaned)
    output: list[str] = []
    stop_words = {
        "despite",
        "the",
        "there",
        "when",
        "on",
        "in",
        "source",
        "update",
        "lore",
        "event",
        "partner",
        "see",
        "also",
    }
    for piece in pieces:
        words = piece.split()
        truncated: list[str] = []
        for index, word in enumerate(words):
            token = re.sub(r"[^A-Za-z]+", "", word).lower()
            if token in stop_words:
                break
            truncated.append(word)
        normalized = _normalize_skin_name(" ".join(truncated), entity_name)
        if normalized:
            output.append(normalized)
    return output


def _clean_skin_source_name(value: str, entity_type: str) -> str:
    cleaned = unquote(str(value))
    cleaned = cleaned.rsplit("/", 1)[-1]
    cleaned = cleaned.split("?", 1)[0]
    cleaned = re.sub(r"\.(png|jpe?g|webp|gif)$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    prefixes = (
        f"{entity_type.title()} 3D ",
        f"{entity_type.title()} ",
        "Weapon 3D ",
        "Weapon ",
        "Tool 3D ",
        "Tool ",
        "Consumable 3D ",
        "Consumable ",
        "Model ",
        "DLC ",
        "Promo ",
    )
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
            break
    return cleaned


def _normalize_skin_name(value: str, entity_name: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return ""
    if cleaned[0].islower() or ")" in cleaned:
        return ""
    cleaned = re.sub(r"\.{2,}$", "", cleaned).strip()
    cleaned = re.sub(r"\b(DLC|Concept)?\s*Art\b$", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(Model|Promo|Render|Screenshot)\b$", " ", cleaned, flags=re.IGNORECASE)
    cleaned = _strip_entity_name_prefix(_clean_text(cleaned), entity_name)
    cleaned = _clean_text(cleaned)
    if not cleaned:
        return ""
    if cleaned.isupper() and len(cleaned) > 1:
        return ""
    key = _slug(cleaned)
    entity_key = _slug(entity_name)
    if key == entity_key:
        return ""
    if key in {"base", "base-weapon", "base-tool", "base-consumable", "weapon", "tool", "consumable"}:
        return ""
    if len(cleaned.split()) > 5:
        return ""
    if len(cleaned) < 3:
        return ""
    return cleaned


def _strip_entity_name_prefix(value: str, entity_name: str) -> str:
    value_words = re.findall(r"[a-z0-9]+|&", value.lower())
    entity_words = re.findall(r"[a-z0-9]+|&", entity_name.lower())
    if not value_words or not entity_words or value_words[: len(entity_words)] != entity_words:
        return value
    original_words = value.split()
    entity_word_count = len(re.sub(r"[_-]+", " ", entity_name).split())
    return " ".join(original_words[entity_word_count:])


def _key_values_from_rows(rows: Sequence[Sequence[str]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in rows:
        if len(row) != 2:
            continue
        key, value = (_clean_text(row[0]).rstrip(":"), _clean_text(row[1]))
        if not key or not value or len(key) > 80:
            continue
        values.setdefault(key, value[:500])
    return values


def _dedupe_images(images: Iterable[WikiImage]) -> list[WikiImage]:
    seen: set[str] = set()
    output: list[WikiImage] = []
    for image in images:
        if image.url in seen:
            continue
        seen.add(image.url)
        output.append(image)
    return output


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _extract_revision_id(html: str) -> str:
    match = re.search(r'"wgRevisionId"\s*:\s*(\d+)', html)
    return match.group(1) if match else ""


def _extract_categories(html: str) -> list[str]:
    match = re.search(r'"wgCategories"\s*:\s*(\[[^\]]*\])', html)
    if not match:
        return []
    try:
        loaded = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in loaded if str(item).strip()]


def _is_ignored_content_class(class_name: str) -> bool:
    classes = set(class_name.lower().split())
    ignored = {
        "toc",
        "mw-editsection",
        "reference",
        "references",
        "navbox",
        "metadata",
        "ambox",
        "catlinks",
        "printfooter",
        "noprint",
    }
    return bool(classes & ignored)


def _page_url(base_url: str, page_name: str) -> str:
    return f"{base_url.rstrip('/')}/wiki/{page_name}"


def _page_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    return unquote(parsed.path.removeprefix("/wiki/"))


def _load_cached_pages(config: HuntWikiPackConfig) -> list[WikiPage]:
    path = _crawl_cache_path(config.output_dir)
    if not config.reuse_crawl_cache or config.refresh or not path.exists():
        return []
    if not _crawl_state_matches(config):
        return []
    pages: list[WikiPage] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pages.append(_wiki_page_from_row(json.loads(line)))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return []
    if config.max_pages > 0 and len(pages) > config.max_pages:
        return pages[: config.max_pages]
    return pages


def _write_crawl_cache(output_dir: Path, pages: Sequence[WikiPage]) -> None:
    path = _crawl_cache_path(output_dir)
    _write_jsonl(path, (asdict(page) for page in pages))


def _append_crawl_cache_page(output_dir: Path, page: WikiPage) -> None:
    path = _crawl_cache_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(page), ensure_ascii=False, sort_keys=True) + "\n")


def _reset_crawl_cache(output_dir: Path) -> None:
    for path in (_crawl_cache_path(output_dir), _crawl_state_path(output_dir)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_crawl_state(config: HuntWikiPackConfig, pages: Sequence[WikiPage], *, complete: bool) -> None:
    payload = {
        "complete": complete,
        "fingerprint": _crawl_cache_fingerprint(config),
        "page_count": len(pages),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _crawl_state_path(config.output_dir).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _crawl_cache_is_complete(config: HuntWikiPackConfig, pages: Sequence[WikiPage]) -> bool:
    state = _read_crawl_state(config)
    return bool(
        state.get("complete")
        and state.get("fingerprint") == _crawl_cache_fingerprint(config)
        and int(state.get("page_count", -1)) == len(pages)
    )


def _crawl_state_matches(config: HuntWikiPackConfig) -> bool:
    state = _read_crawl_state(config)
    if not state:
        return False
    return state.get("fingerprint") == _crawl_cache_fingerprint(config)


def _read_crawl_state(config: HuntWikiPackConfig) -> dict[str, object]:
    path = _crawl_state_path(config.output_dir)
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _crawl_cache_fingerprint(config: HuntWikiPackConfig) -> dict[str, object]:
    return {
        "base_url": config.base_url,
        "seeds": list(config.seeds),
        "allowed_entity_types": list(config.allowed_entity_types),
        "max_pages": config.max_pages,
        "max_depth": config.max_depth,
    }


def _crawl_cache_path(output_dir: Path) -> Path:
    return output_dir / "crawl_pages.jsonl"


def _crawl_state_path(output_dir: Path) -> Path:
    return output_dir / "crawl_state.json"


def _wiki_page_from_row(row: dict[str, object]) -> WikiPage:
    images = [
        WikiImage(
            url=str(image.get("url", "")),
            alt=str(image.get("alt", "")),
            width=_parse_optional_int(str(image.get("width"))) if image.get("width") is not None else None,
            height=_parse_optional_int(str(image.get("height"))) if image.get("height") is not None else None,
        )
        for image in row.get("images", [])
        if isinstance(image, dict)
    ]
    return WikiPage(
        title=str(row.get("title", "")),
        page_name=str(row.get("page_name", "")),
        url=str(row.get("url", "")),
        description=str(row.get("description", "")),
        categories=[str(value) for value in row.get("categories", []) if str(value).strip()],
        revision_id=str(row.get("revision_id", "")),
        text=str(row.get("text", "")),
        key_values={str(key): str(value) for key, value in dict(row.get("key_values", {})).items()},
        links=[str(value) for value in row.get("links", []) if str(value).strip()],
        images=images,
        crawl_depth=_parse_optional_int(str(row.get("crawl_depth"))) or 0,
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_text(value: str) -> str:
    value = unescape(value or "")
    value = re.sub(r"\[\s*edit\s*\]", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\[[0-9]+\]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = _clean_text(str(value))
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return output


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def _embedding_model_path(settings: AppSettings) -> Path:
    return settings.models_dir / "hub"


def _runtime_backend(settings: AppSettings) -> str:
    if settings.gpu_backend == "macos-metal":
        return "metal"
    if settings.gpu_backend in {"cuda", "rocm", "cpu"}:
        return settings.gpu_backend
    if settings.runtime_profile == "macos":
        return "metal"
    return "cpu"


def _embedding_precision(settings: AppSettings) -> str:
    return settings.tier.multimodal_retrieval_precision
