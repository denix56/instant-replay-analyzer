from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
try:
    from datetime import UTC, datetime
except ImportError:  # Python < 3.11 compatibility for remote utility runs.
    from datetime import datetime, timezone

    UTC = timezone.utc
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.knowledge.hunt_wiki_pack import (  # noqa: E402
    HUNT_WIKI_BASE_URL,
    HUNT_WIKI_LICENSE,
    HUNT_WIKI_LICENSE_URL,
    _entity_from_page,
    parse_wiki_page,
)


USER_AGENT = "InstantReplayAnalyzerKnowledgePack/0.1 (+local user initiated)"
API_URL = f"{HUNT_WIKI_BASE_URL}/api.php"
OUTPUT_PATH = ROOT / "backend" / "app" / "knowledge" / "hunt_weapon_skin_map.json"


def main() -> int:
    titles = _weapon_titles()
    skins: dict[tuple[str, str], dict[str, str]] = {}
    for index, title in enumerate(titles, start=1):
        page_url = f"{HUNT_WIKI_BASE_URL}/wiki/{title.replace(' ', '_')}"
        html = _parse_page_html(title)
        page = parse_wiki_page(html, page_url)
        entity = _entity_from_page(page)
        if entity.type != "weapon":
            continue
        for skin in _skins_from_key_values(entity.key_values):
            skins[(skin.lower(), entity.name.lower())] = {
                "skin": skin,
                "weapon": entity.name,
                "display_name": f"{entity.name} ({skin} skin)",
                "source_url": entity.source_url,
            }
        if index % 25 == 0:
            print(f"processed {index}/{len(titles)} weapon pages", file=sys.stderr)
        time.sleep(0.03)

    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": f"{HUNT_WIKI_BASE_URL}/wiki/Category:Weapons via MediaWiki API",
        "license": HUNT_WIKI_LICENSE,
        "license_url": HUNT_WIKI_LICENSE_URL,
        "skins": sorted(skins.values(), key=lambda row: (row["skin"].lower(), row["weapon"].lower())),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['skins'])} skin mappings to {OUTPUT_PATH}")
    return 0


def _weapon_titles() -> list[str]:
    titles: list[str] = []
    params: dict[str, str] = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:Weapons",
        "cmlimit": "max",
        "format": "json",
    }
    while True:
        data = _api(params)
        for row in data.get("query", {}).get("categorymembers", []):
            title = str(row.get("title") or "")
            if title.startswith("Weapons/"):
                titles.append(title)
        continuation = data.get("continue", {})
        if "cmcontinue" not in continuation:
            break
        params["cmcontinue"] = str(continuation["cmcontinue"])
    return sorted(set(titles))


def _parse_page_html(title: str) -> str:
    data = _api(
        {
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
            "formatversion": "2",
        }
    )
    parsed = data.get("parse", {})
    html = parsed.get("text")
    if not isinstance(html, str):
        raise RuntimeError(f"Wiki API parse response did not include HTML for {title!r}.")
    return html


def _api(params: dict[str, str]) -> dict[str, Any]:
    url = API_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Unexpected wiki API response for {url}")
    return loaded


def _skins_from_key_values(key_values: dict[str, str]) -> list[str]:
    raw = key_values.get("Skins") or ""
    skins: list[str] = []
    seen: set[str] = set()
    for value in raw.split(","):
        cleaned = " ".join(value.strip().split())
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        skins.append(cleaned)
    return skins


if __name__ == "__main__":
    raise SystemExit(main())
