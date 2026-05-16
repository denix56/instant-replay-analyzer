import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app import pipeline
from backend.app.config import AppSettings
from backend.app.embeddings.hf_multimodal_embedder import EmbeddingConfig, HuggingFaceMultimodalEmbedder
from backend.app.embeddings.vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreConfig,
    VectorStore,
    VectorStoreConfig,
    VectorSearchHit,
)
from backend.app.llm.schemas import ClipMetadata, ClipRecord, SearchRequest
from backend.app.models import SearchResult as ApiSearchResult
from backend.app.search.query import ClipSearchService
from backend.app.db import Database
from backend.app.pipeline import (
    _boost_hud_matches,
    _boost_player_kill_intent,
    _filter_search_results_by_threshold,
    _hit_to_result,
    _is_first_hud_window_segment,
    _is_hud_detection_window_segment,
    _is_player_kill_clip,
    _sqlite_search,
)


def _clip(embedder, clip_id, text, tags=None, game="Arena"):
    return ClipRecord(
        clip_id=clip_id,
        embedding=embedder.embed_text(text),
        metadata=ClipMetadata(
            clip_id=clip_id,
            path=f"{clip_id}.mp4",
            title=text,
            game=game,
            tags=tags or [],
        ),
        transcript=text,
        summary=text,
    )


def _search_settings_with_clip(tmp_path):
    settings = AppSettings(
        clips_dir=tmp_path / "clips",
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        allow_mock_models=True,
        auto_download_models=False,
        qdrant_url="local",
        search_min_score=0.0,
    )
    settings.ensure_dirs()
    db = Database(settings.db_path)
    try:
        clip_id = db.upsert_clip(
            {
                "filename": "clip.mp4",
                "path": "clip.mp4",
                "relative_path": "clip.mp4",
                "group_name": "Hunt",
                "summary": "A fight near the clue.",
                "duration": 25.0,
            }
        )
    finally:
        db.close()
    return settings, clip_id


def _patch_search_dependencies(monkeypatch, settings, clip_id):
    class FakeEmbedder:
        dimension = 3

        def embed_query(self, _query):
            return [1.0, 0.0, 0.0]

    class FakeVectorStore:
        using_qdrant = False

        def search(self, collection, _query_vector, _top_k, _filters):
            if collection != "summary":
                return []
            return [
                VectorSearchHit(
                    "summary-clip",
                    0.75,
                    {
                        "clip_id": clip_id,
                        "file_name": "clip.mp4",
                        "field": "summary",
                        "collection_kind": "summary",
                    },
                )
            ]

    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "_embedder", lambda _settings: FakeEmbedder())
    monkeypatch.setattr(pipeline, "_vector_store", lambda _settings, _dimension: FakeVectorStore())


def test_search_service_returns_semantic_and_ranked_result():
    embedder = HuggingFaceMultimodalEmbedder(EmbeddingConfig(dimension=32))
    store = VectorStore(VectorStoreConfig(dimension=32))
    service = ClipSearchService(embedder=embedder, vector_store=store)
    service.index_clips(
        [
            _clip(embedder, "loss", "slow rotation and looting", tags=["loot"]),
            _clip(embedder, "win", "clutch final kill for the win", tags=["win", "kill"]),
        ]
    )

    results = service.search("final kill")

    assert results[0].clip_id == "win"
    assert results[0].score >= results[-1].score


def test_search_service_filters_by_game_and_tags():
    embedder = HuggingFaceMultimodalEmbedder(EmbeddingConfig(dimension=32))
    store = VectorStore(VectorStoreConfig(dimension=32))
    service = ClipSearchService(embedder=embedder, vector_store=store)
    service.index_clips(
        [
            _clip(embedder, "arena", "final kill", tags=["win"], game="Arena"),
            _clip(embedder, "other", "final kill", tags=["win"], game="Other"),
        ]
    )

    results = service.search(SearchRequest(query="final kill", game="Arena", tags=["win"]))

    assert [result.clip_id for result in results] == ["arena"]


def test_multimodal_embedder_accepts_audio_and_image_paths(tmp_path):
    image = tmp_path / "frame.jpg"
    audio = tmp_path / "audio.wav"
    image.write_bytes(b"fake-image")
    audio.write_bytes(b"fake-audio")

    embedder = HuggingFaceMultimodalEmbedder(EmbeddingConfig(dimension=16))
    vector = embedder.embed_multimodal(text="boss lair fight", image_path=image, audio_path=audio)

    assert len(vector) == 16
    assert any(value != 0 for value in vector)


def test_hud_detection_window_uses_only_segments_inside_first_15_seconds():
    assert _is_first_hud_window_segment({"start_time": 0.0, "end_time": 2.0})
    assert _is_first_hud_window_segment({"start_time": 13.0, "end_time": 15.0})
    assert not _is_first_hud_window_segment({"start_time": 15.0, "end_time": 17.0})
    assert not _is_first_hud_window_segment({"start_time": 14.0, "end_time": 16.0})


def test_hud_detection_window_uses_18_to_20_seconds_for_kill_clips():
    assert _is_player_kill_clip({"filename": "Hunt Showdown 23.22.40.24.Hunter killed.DVR.mp4"})
    assert not _is_player_kill_clip({"filename": "Hunt Showdown 22.43.24.Player downed.DVR.mp4"})
    assert _is_hud_detection_window_segment({"start_time": 18.0, "end_time": 20.0}, kill_clip=True)
    assert _is_hud_detection_window_segment({"start_time": 17.5, "end_time": 19.5}, kill_clip=True)
    assert not _is_hud_detection_window_segment({"start_time": 16.0, "end_time": 18.0}, kill_clip=True)
    assert not _is_hud_detection_window_segment({"start_time": 20.0, "end_time": 22.0}, kill_clip=True)


def test_qdrant_store_uses_query_points_client_api():
    class FakeHit:
        id = "qdrant-id"
        score = 0.91
        payload = {"vector_store_key": "clip-1-segment-1", "clip_id": 1}

    class FakeResponse:
        points = [FakeHit()]

    class FakeClient:
        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name="av_segments")])

        def query_points(self, **kwargs):
            self.kwargs = kwargs
            return FakeResponse()

    client = FakeClient()
    store = QdrantVectorStore(QdrantVectorStoreConfig(dimension=3, prefer_qdrant=False))
    store._client = client

    hits = store.search("av_segments", [1.0, 0.0, 0.0], top_k=1)

    assert hits[0].point_id == "clip-1-segment-1"
    assert hits[0].score == 0.91
    assert client.kwargs["query"] == [1.0, 0.0, 0.0]


def test_qdrant_store_embedded_local_mode(tmp_path):
    store = QdrantVectorStore(
        QdrantVectorStoreConfig(url="local", local_path=str(tmp_path / "qdrant"), dimension=3)
    )

    store.add_vector(
        "av_segments",
        "clip-1-segment-1",
        [1.0, 0.0, 0.0],
        {"clip_id": 1, "group_name": "Ungrouped"},
    )
    hits = store.search("av_segments", [1.0, 0.0, 0.0], top_k=1)

    assert store.using_qdrant is True
    assert hits[0].point_id == "clip-1-segment-1"
    assert hits[0].payload["clip_id"] == 1


def test_pipeline_vector_store_fallback_tracks_mock_mode(tmp_path, monkeypatch):
    captured = []

    class FakeQdrantVectorStore:
        def __init__(self, config):
            captured.append(config)

    monkeypatch.setattr(pipeline, "QdrantVectorStore", FakeQdrantVectorStore)

    strict_settings = AppSettings(data_dir=tmp_path / "strict", models_dir=tmp_path / "models", allow_mock_models=False)
    dev_settings = AppSettings(data_dir=tmp_path / "dev", models_dir=tmp_path / "models", allow_mock_models=True)

    pipeline._vector_store(strict_settings, 3)
    pipeline._vector_store(dev_settings, 3)

    assert captured[0].mock_fallback is False
    assert captured[1].mock_fallback is True


def test_run_search_uses_reranker_by_default(tmp_path, monkeypatch):
    settings, clip_id = _search_settings_with_clip(tmp_path)
    _patch_search_dependencies(monkeypatch, settings, clip_id)
    calls = []

    def fake_rerank(_settings, query, results):
        calls.append((query, len(results)))
        for result in results:
            result.score = 0.99
        return SimpleNamespace(results=results, warning=None)

    monkeypatch.setattr(pipeline, "_rerank_search_results", fake_rerank)

    response = pipeline.run_search(params={"query": "fight near clue", "top_k": 1, "min_score": 0.0})

    assert calls == [("fight near clue", 1)]
    assert response["results"][0]["score"] == 0.99


def test_run_search_can_disable_reranking_per_request(tmp_path, monkeypatch):
    settings, clip_id = _search_settings_with_clip(tmp_path)
    _patch_search_dependencies(monkeypatch, settings, clip_id)
    monkeypatch.setattr(
        pipeline,
        "_rerank_search_results",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reranker should not be called")),
    )

    response = pipeline.run_search(
        params={"query": "fight near clue", "top_k": 1, "min_score": 0.0, "enable_reranking": False}
    )

    assert response["results"][0]["score"] == 0.2


def test_hud_loadout_matches_are_added_to_search_hits(tmp_path):
    db = Database(tmp_path / "app.db")
    try:
        clip_id = db.upsert_clip({"filename": "dolch.mp4", "path": "dolch.mp4", "group_name": "Hunt"})
        segment_id = db.upsert_segment(
            {
                "clip_id": clip_id,
                "group_name": "Hunt",
                "start_time": 4.0,
                "end_time": 6.0,
                "duration": 2.0,
                "modality": "video_only",
                "segment_settings_hash": "settings",
            }
        )
        db.replace_hud_detections(
            clip_id,
            segment_id,
            [
                {
                    "frame_path": "frame.jpg",
                    "timestamp": 4.0,
                    "slot_key": "1",
                    "is_active": 1,
                    "entity_id": "weapon:dolch-96",
                    "entity_name": "Dolch 96",
                    "entity_type": "weapon",
                    "confidence": 0.9,
                }
            ],
        )

        hits = _boost_hud_matches(db, "dolch", [], top=10, group_name="Hunt")

        assert hits[0][1]["clip_id"] == clip_id
        assert hits[0][1]["collection_kind"] == "hud_loadout"
    finally:
        db.close()


def test_hud_loadout_search_handles_compact_weapon_name(tmp_path):
    db = Database(tmp_path / "app.db")
    try:
        clip_id = db.upsert_clip({"filename": "auto5.mp4", "path": "auto5.mp4", "group_name": "Hunt"})
        segment_id = db.upsert_segment(
            {
                "clip_id": clip_id,
                "group_name": "Hunt",
                "start_time": 18.0,
                "end_time": 20.0,
                "duration": 2.0,
                "modality": "audio_video",
                "segment_settings_hash": "settings",
            }
        )
        db.replace_hud_detections(
            clip_id,
            segment_id,
            [
                {
                    "frame_path": "frame.jpg",
                    "timestamp": 18.0,
                    "slot_key": "current_ocr",
                    "is_active": 1,
                    "entity_id": "weapon:auto-5",
                    "entity_name": "Auto-5",
                    "entity_type": "weapon",
                    "confidence": 0.9,
                }
            ],
        )

        hits = _boost_hud_matches(db, "auto5", [], top=10, group_name="Hunt")

        assert hits[0][1]["clip_id"] == clip_id
        assert hits[0][1]["text"] == "Auto-5"
    finally:
        db.close()


def test_player_kill_intent_prefers_hunter_killed_clip_over_player_death(tmp_path):
    db = Database(tmp_path / "app.db")
    try:
        kill_clip = db.upsert_clip(
            {
                "filename": "Hunt Showdown 23.23.53.25.Hunter killed.DVR.mp4",
                "path": "kill.mp4",
                "group_name": "Hunt",
                "summary": "You used Auto-5 near a window.",
            }
        )
        db.upsert_segment(
            {
                "clip_id": kill_clip,
                "group_name": "Hunt",
                "start_time": 18.0,
                "end_time": 20.0,
                "duration": 2.0,
                "modality": "audio_video",
                "representative_frame_path": "kill-frame.jpg",
                "segment_settings_hash": "settings",
            }
        )
        db.replace_hud_detections(
            kill_clip,
            db.list_segments(kill_clip)[0]["id"],
            [
                {
                    "frame_path": "kill-frame.jpg",
                    "timestamp": 18.0,
                    "slot_key": "current_ocr",
                    "is_active": 1,
                    "entity_id": "weapon:auto-5",
                    "entity_name": "Auto-5",
                    "entity_type": "weapon",
                    "confidence": 0.9,
                }
            ],
        )
        death_clip = db.upsert_clip(
            {
                "filename": "Hunt Showdown 22.43.24.Player downed.DVR.mp4",
                "path": "death.mp4",
                "group_name": "Hunt",
                "summary": "You were downed near a window with Auto-5 equipped.",
            }
        )
        death_segment = db.upsert_segment(
            {
                "clip_id": death_clip,
                "group_name": "Hunt",
                "start_time": 18.0,
                "end_time": 20.0,
                "duration": 2.0,
                "modality": "audio_video",
                "segment_settings_hash": "settings",
            }
        )
        db.replace_death_screen_detection(
            death_clip,
            death_segment,
            {
                "frame_path": "death-frame.jpg",
                "timestamp": 20.0,
                "status": "downed",
                "killed_with": "Hunting Bow",
                "killer_name": "Cain",
                "raw_text": "",
                "confidence": 0.9,
            },
        )

        ranked = _boost_player_kill_intent(
            db,
            "i kill with auto5 near the window",
            [
                (4.0, {"clip_id": death_clip, "collection_kind": "metadata", "modality": "metadata"}),
                (0.0, {"clip_id": kill_clip, "collection_kind": "metadata", "modality": "metadata"}),
            ],
            top=10,
            group_name="Hunt",
        )

        assert ranked[0][1]["clip_id"] == kill_clip
        assert ranked[0][1]["collection_kind"] == "player_kill_intent"
        assert ranked[0][1]["start_time"] == 18.0
        assert ranked[0][1]["end_time"] == 20.0
    finally:
        db.close()


def test_hit_to_result_preserves_av_segment_timestamp(tmp_path):
    db = Database(tmp_path / "app.db")
    try:
        clip_id = db.upsert_clip({"filename": "av.mp4", "path": "av.mp4", "group_name": "Arena"})
        db.upsert_segment(
            {
                "clip_id": clip_id,
                "group_name": "Arena",
                "start_time": 12.0,
                "end_time": 14.5,
                "duration": 2.5,
                "modality": "audio_video",
                "representative_frame_path": "frame-12.jpg",
                "segment_settings_hash": "settings",
            }
        )

        result = _hit_to_result(
            db,
            0.9,
            {
                "clip_id": clip_id,
                "collection_kind": "av_segments",
                "modality": "audio_video",
                "start_time": 12.0,
                "end_time": 14.5,
                "representative_frame_path": "frame-12.jpg",
            },
        )

        assert result is not None
        assert result.best_timestamp == 12.0
        assert result.segment_start == 12.0
        assert result.segment_end == 14.5
        assert result.preview_frame == "frame-12.jpg"
    finally:
        db.close()


def test_transcript_hit_uses_transcript_time_and_nearest_av_segment(tmp_path):
    db = Database(tmp_path / "app.db")
    try:
        clip_id = db.upsert_clip({"filename": "transcript.mp4", "path": "transcript.mp4", "group_name": "Arena"})
        db.upsert_segment(
            {
                "clip_id": clip_id,
                "group_name": "Arena",
                "start_time": 8.0,
                "end_time": 12.0,
                "duration": 4.0,
                "modality": "audio_video",
                "representative_frame_path": "frame-8.jpg",
                "segment_settings_hash": "settings",
            }
        )
        db.add_transcript({"clip_id": clip_id, "start_time": 10.0, "end_time": 11.0, "text": "final kill callout"})

        result = _hit_to_result(
            db,
            0.8,
            {"clip_id": clip_id, "collection_kind": "transcript_text", "modality": "transcript", "text": "final kill callout"},
            query="final kill",
        )

        assert result is not None
        assert result.best_timestamp == 10.0
        assert result.segment_start == 8.0
        assert result.segment_end == 12.0
        assert result.preview_frame == "frame-8.jpg"
    finally:
        db.close()


def test_sqlite_fallback_uses_transcript_time_when_available(tmp_path):
    db = Database(tmp_path / "app.db")
    try:
        clip_id = db.upsert_clip(
            {"filename": "sqlite.mp4", "path": "sqlite.mp4", "group_name": "Arena", "summary": "quiet rotation"}
        )
        db.upsert_segment(
            {
                "clip_id": clip_id,
                "group_name": "Arena",
                "start_time": 20.0,
                "end_time": 24.0,
                "duration": 4.0,
                "modality": "audio_video",
                "segment_settings_hash": "settings",
            }
        )
        db.add_transcript({"clip_id": clip_id, "start_time": 22.0, "end_time": 23.0, "text": "rotate now"})

        results = _sqlite_search(db, "rotate", 10, "Arena")

        assert len(results) == 1
        assert results[0].clip_id == clip_id
        assert results[0].best_timestamp == 22.0
        assert results[0].segment_start == 20.0
        assert results[0].segment_end == 24.0
    finally:
        db.close()


def test_sqlite_fallback_preserves_missing_timestamp_when_no_timing_exists(tmp_path):
    db = Database(tmp_path / "app.db")
    try:
        clip_id = db.upsert_clip(
            {"filename": "untimed.mp4", "path": "untimed.mp4", "group_name": "Arena", "summary": "rotate now"}
        )

        results = _sqlite_search(db, "rotate", 10, "Arena")

        assert len(results) == 1
        assert results[0].clip_id == clip_id
        assert results[0].best_timestamp is None
        assert results[0].segment_start is None
        assert results[0].segment_end is None
    finally:
        db.close()


def test_search_threshold_returns_all_sorted_results_above_cutoff():
    results = [
        ApiSearchResult(clip_id=1, clip_filename="low.mp4", source_path="low.mp4", group_name="Hunt", score=0.34),
        ApiSearchResult(clip_id=2, clip_filename="high.mp4", source_path="high.mp4", group_name="Hunt", score=0.91),
        ApiSearchResult(clip_id=3, clip_filename="mid.mp4", source_path="mid.mp4", group_name="Hunt", score=0.52),
    ]

    filtered = _filter_search_results_by_threshold(results, min_score=0.35)

    assert [result.clip_id for result in filtered] == [2, 3]


def test_hud_boost_preserves_existing_av_timestamp(tmp_path):
    db = Database(tmp_path / "app.db")
    try:
        clip_id = db.upsert_clip({"filename": "boost.mp4", "path": "boost.mp4", "group_name": "Hunt"})
        segment_id = db.upsert_segment(
            {
                "clip_id": clip_id,
                "group_name": "Hunt",
                "start_time": 4.0,
                "end_time": 6.0,
                "duration": 2.0,
                "modality": "video_only",
                "segment_settings_hash": "settings",
            }
        )
        db.replace_hud_detections(
            clip_id,
            segment_id,
            [
                {
                    "frame_path": "frame.jpg",
                    "timestamp": 4.0,
                    "slot_key": "1",
                    "is_active": 1,
                    "entity_id": "weapon:dolch-96",
                    "entity_name": "Dolch 96",
                    "entity_type": "weapon",
                    "confidence": 0.9,
                }
            ],
        )

        hits = _boost_hud_matches(
            db,
            "dolch",
            [
                (
                    0.5,
                    {
                        "clip_id": clip_id,
                        "collection_kind": "av_segments",
                        "modality": "video_only",
                        "start_time": 10.0,
                        "end_time": 12.0,
                    },
                )
            ],
            top=10,
            group_name="Hunt",
        )

        assert hits[0][1]["start_time"] == 10.0
        assert hits[0][1]["end_time"] == 12.0
    finally:
        db.close()
