FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/workspace/data \
    MODELS_DIR=/workspace/models \
    ALLOW_MOCK_MODELS=true \
    AUTO_DOWNLOAD_MODELS=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends chromium chromium-driver ca-certificates fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY backend/requirements-crawler.txt /tmp/requirements-crawler.txt
RUN pip install --no-cache-dir -r /tmp/requirements-crawler.txt

COPY backend /workspace/backend

ENTRYPOINT ["python", "-m", "backend.app.cli", "build-hunt-knowledge-pack"]
CMD ["--output", "/workspace/data/packs/hunt-knowledge-pack", "--max-pages", "0", "--max-depth", "4", "--delay", "3.0", "--crawl-concurrency", "1", "--selenium-fetch", "--max-images-per-page", "32", "--allow-mock-embeddings"]
