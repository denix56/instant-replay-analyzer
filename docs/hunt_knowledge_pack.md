# Hunt Knowledge Pack

The backend can build a redistributable Hunt: Showdown 1896 knowledge pack from allowed
wiki pages on `huntshowdown.wiki.gg`.

The pack contains normalized entity records, searchable text chunks, Hugging Face
Qwen3-VL text embeddings, wiki article images, and attribution metadata:

```text
manifest.json
entities.jsonl
chunks.jsonl
embedding_index.jsonl
embeddings.npy
media_index.jsonl
media/images/
attribution.jsonl
```

Build a release pack with the configured local embedding runtime:

```bash
python -m app build-hunt-knowledge-pack \
  --output data/packs/hunt-knowledge-pack \
  --refresh
```

Image scraping is enabled by default. Use `--no-images` for a text-only pack, or tune
`--max-images-per-page` if a release pack should keep fewer reference images.
Progress is printed to stderr during crawling, image downloads, embedding, and pack writes;
use `--quiet` to disable it.

The builder writes `crawl_pages.jsonl` immediately after a successful crawl and reuses it
on later runs from the same output directory. Use `--refresh` to force a clean rebuild or
`--no-crawl-cache` to ignore only the crawl cache.

Page fetching runs with bounded parallelism. The default is `--crawl-concurrency 2`;
set it to `1` for strictly serial crawling. The crawler keeps at least a 0.5 second
gap between scheduled wiki page requests, and retries transient `403` responses before
skipping an inaccessible page.

If plain HTTP requests are blocked while the site still works in a browser, add
`--browser-fetch`. That uses headless Chromium for article HTML and forces page crawling
to serial mode while still reusing `crawl_pages.jsonl`.

Use `--max-pages 0 --max-depth -1` for an unbounded crawl, but keep the default request
delay or a larger one. The crawler uses normal `/wiki/...` pages and does not use
`api.php`, because wiki.gg disallows crawling that endpoint in `robots.txt`.

For tests or offline development only:

```bash
python -m app build-hunt-knowledge-pack \
  --output data/packs/hunt-knowledge-pack-dev \
  --max-pages 25 \
  --allow-mock-embeddings \
  --refresh
```

The generated content is derived from the Hunt wiki and includes source URLs plus CC BY-SA
4.0 attribution metadata. Do not use it for model training; use it as a local search/RAG
index.
