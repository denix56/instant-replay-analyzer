# Hunt Wiki Crawler Docker Runner

Build:

```bash
docker build -f docker/hunt-crawler.Dockerfile -t instant-replay-hunt-crawler .
```

Resume the local crawl cache and write into the existing pack directory:

```bash
docker run --rm -it \
  -v "$PWD/data:/workspace/data" \
  instant-replay-hunt-crawler \
  --output /workspace/data/packs/hunt-knowledge-pack \
  --max-pages 0 \
  --max-depth 4 \
  --delay 3.0 \
  --crawl-concurrency 1 \
  --selenium-fetch \
  --max-images-per-page 32 \
  --allow-mock-embeddings
```

Use a custom Selenium driver factory:

```bash
docker run --rm -it \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/local_driver_hooks:/workspace/local_driver_hooks" \
  -e PYTHONPATH=/workspace:/workspace/local_driver_hooks \
  -e HUNT_WIKI_SELENIUM_DRIVER_FACTORY=my_driver:create_driver \
  instant-replay-hunt-crawler \
  --output /workspace/data/packs/hunt-knowledge-pack \
  --max-pages 0 \
  --max-depth 4 \
  --delay 3.0 \
  --crawl-concurrency 1 \
  --selenium-fetch \
  --max-images-per-page 32 \
  --allow-mock-embeddings
```

The custom factory must be a callable that accepts `HuntWikiPackConfig` and returns a Selenium-compatible WebDriver. Keep the returned object compatible with `get`, `page_source`, `current_url`, `execute_script`, and `quit`.
