from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

from .config import get_settings
from .operations.events import model_dump
from .operations.manager import OperationManager
from .operations.schemas import AppConfig, OperationState


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _print_json(value: Any) -> None:
    print(json.dumps(value, default=_json_default, indent=2, sort_keys=True))


async def _run_operation(kind: str, params: Dict[str, Any]) -> int:
    manager = OperationManager()
    status = await manager.start(kind, params)
    final_status = await manager.wait(status.operation_id)
    _print_json(model_dump(final_status))
    return 0 if final_status.state == OperationState.SUCCEEDED else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="instant-replay-analyzer",
        description="Local-first semantic gameplay clip search API and operations runner.",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="Run the FastAPI server.")
    serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    serve.add_argument("--port", default=8000, type=int, help="Port to bind.")
    serve.add_argument("--log-level", default="info", help="Uvicorn log level.")

    index = subparsers.add_parser("index", help="Index replay clips into the semantic search store.")
    index.add_argument("--source", default=None, help="Replay file or directory to index.")
    index.add_argument("--replay-dir", default=None, help="Directory containing replay captures.")
    index.add_argument("--force", action="store_true", help="Rebuild existing index entries.")

    search = subparsers.add_parser("search", help="Search clips by semantic gameplay text.")
    search.add_argument("query", help="Natural language gameplay query.")
    search.add_argument("--limit", default=10, type=int, help="Maximum number of clips to return.")

    analyze = subparsers.add_parser("analyze", help="Analyze a selected clip with the configured reasoning path.")
    analyze.add_argument("--clip-id", required=True, help="Clip ID to analyze.")

    download = subparsers.add_parser("download-models", help="Download configured Hugging Face models.")
    download.add_argument("--tier", default=None, choices=["default", "quality"], help="Model tier to download.")
    download.add_argument("--models-dir", default=None, help="Target models directory.")
    download.add_argument("--force", action="store_true", help="Force re-download existing snapshots.")
    download.add_argument("--dry-run", action="store_true", help="Report missing/present snapshots without downloading.")

    hunt_pack = subparsers.add_parser(
        "build-hunt-knowledge-pack",
        help="Build a redistributable Hunt wiki text knowledge pack with local embeddings.",
    )
    hunt_pack.add_argument("--output", default=None, help="Output directory for the generated pack.")
    hunt_pack.add_argument("--max-pages", default=750, type=int, help="Maximum wiki pages to crawl; 0 means unlimited.")
    hunt_pack.add_argument(
        "--max-depth",
        default=4,
        type=int,
        help="Maximum link depth from seed pages; -1 means unlimited.",
    )
    hunt_pack.add_argument("--delay", default=0.75, type=float, help="Delay between wiki page requests in seconds.")
    hunt_pack.add_argument("--crawl-concurrency", default=2, type=int, help="Parallel wiki page fetch workers.")
    hunt_pack.add_argument("--batch-size", default=16, type=int, help="Embedding batch size.")
    hunt_pack.add_argument("--refresh", action="store_true", help="Delete and rebuild the output directory.")
    hunt_pack.add_argument("--quiet", action="store_true", help="Disable progress output.")
    hunt_pack.add_argument("--no-crawl-cache", action="store_true", help="Do not reuse crawl_pages.jsonl from the output directory.")
    hunt_pack.add_argument("--browser-fetch", action="store_true", help="Fetch wiki article HTML through headless Chromium.")
    hunt_pack.add_argument("--selenium-fetch", action="store_true", help="Fetch wiki article HTML through Selenium Chrome.")
    hunt_pack.add_argument("--selenium-remote-url", default="", help="Remote Selenium WebDriver URL.")
    hunt_pack.add_argument("--selenium-profile-dir", default="", help="Persistent Selenium Chrome profile directory.")
    hunt_pack.add_argument("--selenium-headed", action="store_true", help="Run Selenium Chrome with a visible window.")
    hunt_pack.add_argument(
        "--show-browser-on-block",
        action="store_true",
        help="Use visible Chromium for browser fetches and wait for manual wiki.gg checks on 403/block pages.",
    )
    hunt_pack.add_argument("--no-images", action="store_true", help="Skip wiki image downloads.")
    hunt_pack.add_argument(
        "--max-images-per-page",
        default=12,
        type=int,
        help="Maximum article images to store for each crawled page.",
    )
    hunt_pack.add_argument(
        "--min-image-long-side",
        default=64,
        type=int,
        help="Skip images whose largest known side is below this pixel count.",
    )
    hunt_pack.add_argument(
        "--allow-mock-embeddings",
        action="store_true",
        help="Allow deterministic fallback embeddings when the local embedding runtime cannot run.",
    )
    hunt_pack.add_argument(
        "--seed",
        action="append",
        default=None,
        help="Replacement wiki page seed. Repeat to crawl from multiple pages.",
    )

    subparsers.add_parser("reset-index", help="Delete local SQLite index metadata.")
    subparsers.add_parser("self-test", help="Run a lightweight local operation smoke test.")
    smoke_models = subparsers.add_parser("smoke-models", help="Check local model server health and minimal requests.")
    smoke_models.add_argument("--tier", default=None, choices=["default", "quality"], help="Model tier to smoke test.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "serve":
        from .api.server import run_server

        run_server(AppConfig(host=args.host, port=args.port, log_level=args.log_level))
        return 0

    if args.command == "index":
        params = {"source": args.source, "replay_dir": args.replay_dir, "force": args.force, "metadata": {}}
        return asyncio.run(_run_operation("index", params))

    if args.command == "search":
        params = {"query": args.query, "limit": args.limit, "filters": {}}
        return asyncio.run(_run_operation("search", params))

    if args.command == "analyze":
        params = {"clip_id": args.clip_id}
        return asyncio.run(_run_operation("analyze", params))

    if args.command == "download-models":
        from .model_downloader import ensure_models

        settings = get_settings()
        results = ensure_models(
            args.tier or settings.model_tier,
            args.models_dir or settings.models_dir,
            gpu_backend=settings.gpu_backend,
            force=args.force,
            dry_run=args.dry_run,
        )
        _print_json([result.__dict__ for result in results])
        return 1 if any(result.status == "failed" for result in results) else 0

    if args.command == "build-hunt-knowledge-pack":
        from pathlib import Path

        from .knowledge.hunt_wiki_pack import DEFAULT_SEED_PAGES, HuntWikiPackConfig, build_hunt_knowledge_pack

        settings = get_settings()
        output = Path(args.output) if args.output else settings.data_dir / "packs" / "hunt-knowledge-pack"
        config = HuntWikiPackConfig(
            output_dir=output,
            seeds=tuple(args.seed) if args.seed else DEFAULT_SEED_PAGES,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            delay_seconds=args.delay,
            crawl_concurrency=args.crawl_concurrency,
            batch_size=args.batch_size,
            require_real_embeddings=not args.allow_mock_embeddings,
            include_images=not args.no_images,
            max_images_per_page=args.max_images_per_page,
            min_image_long_side=args.min_image_long_side,
            refresh=args.refresh,
            progress=not args.quiet,
            reuse_crawl_cache=not args.no_crawl_cache,
            browser_fetch=(args.browser_fetch or args.show_browser_on_block) and not args.selenium_fetch,
            selenium_fetch=args.selenium_fetch,
            selenium_remote_url=args.selenium_remote_url,
            selenium_headless=not args.selenium_headed,
            selenium_profile_dir=Path(args.selenium_profile_dir) if args.selenium_profile_dir else None,
            show_browser_on_block=args.show_browser_on_block,
        )
        manifest = build_hunt_knowledge_pack(config)
        _print_json(manifest)
        return 0

    if args.command == "reset-index":
        from .db import Database

        settings = get_settings()
        db = Database(settings.db_path)
        try:
            db.reset_index()
        finally:
            db.close()
        _print_json({"ok": True, "db_path": str(settings.db_path)})
        return 0

    if args.command == "self-test":
        return asyncio.run(_run_operation("search", {"query": "opening fight", "limit": 1, "filters": {}}))

    if args.command == "smoke-models":
        from .runtime_smoke import smoke_models

        settings = get_settings()
        _print_json(smoke_models(args.tier or settings.model_tier, settings))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
