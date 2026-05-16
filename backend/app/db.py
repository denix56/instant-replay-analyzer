from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .models import OperationEvent, OperationRecord, utc_now


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY,
    file_hash TEXT UNIQUE,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    relative_path TEXT,
    source_root TEXT,
    group_name TEXT,
    duration REAL,
    size_bytes INTEGER,
    created_at TEXT,
    modified_at TEXT,
    indexed_at TEXT,
    last_seen_at TEXT,
    width INTEGER,
    height INTEGER,
    fps REAL,
    codec TEXT,
    status TEXT,
    scan_status TEXT,
    summary TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS av_segments (
    id INTEGER PRIMARY KEY,
    clip_id INTEGER NOT NULL,
    group_name TEXT,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    duration REAL NOT NULL,
    modality TEXT NOT NULL,
    representative_frame_path TEXT,
    video_segment_path TEXT,
    audio_segment_path TEXT,
    embedding_id TEXT,
    embedding_model TEXT,
    embedding_precision TEXT,
    runtime_backend TEXT,
    segment_settings_hash TEXT,
    created_at TEXT,
    error_message TEXT,
    FOREIGN KEY (clip_id) REFERENCES clips(id)
);

CREATE TABLE IF NOT EXISTS av_segment_frames (
    id INTEGER PRIMARY KEY,
    segment_id INTEGER NOT NULL,
    frame_path TEXT NOT NULL,
    timestamp REAL NOT NULL,
    frame_index INTEGER,
    FOREIGN KEY (segment_id) REFERENCES av_segments(id)
);

CREATE TABLE IF NOT EXISTS audio_artifacts (
    id INTEGER PRIMARY KEY,
    clip_id INTEGER NOT NULL,
    audio_path TEXT,
    has_audio INTEGER,
    extraction_status TEXT,
    error_message TEXT,
    FOREIGN KEY (clip_id) REFERENCES clips(id)
);

CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY,
    clip_id INTEGER NOT NULL,
    start_time REAL,
    end_time REAL,
    text TEXT NOT NULL,
    confidence REAL,
    model_name TEXT,
    embedding_id TEXT,
    embedding_model TEXT,
    embedding_precision TEXT,
    runtime_backend TEXT,
    FOREIGN KEY (clip_id) REFERENCES clips(id)
);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS clip_tags (
    clip_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    source TEXT,
    PRIMARY KEY (clip_id, tag_id),
    FOREIGN KEY (clip_id) REFERENCES clips(id),
    FOREIGN KEY (tag_id) REFERENCES tags(id)
);

CREATE TABLE IF NOT EXISTS text_items (
    id INTEGER PRIMARY KEY,
    clip_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding_id TEXT,
    embedding_model TEXT,
    embedding_precision TEXT,
    runtime_backend TEXT,
    FOREIGN KEY (clip_id) REFERENCES clips(id)
);

CREATE TABLE IF NOT EXISTS hud_loadout_detections (
    id INTEGER PRIMARY KEY,
    clip_id INTEGER NOT NULL,
    segment_id INTEGER NOT NULL,
    frame_path TEXT NOT NULL,
    timestamp REAL,
    slot_key TEXT NOT NULL,
    is_active INTEGER DEFAULT 0,
    entity_id TEXT,
    entity_name TEXT,
    entity_type TEXT,
    confidence REAL DEFAULT 0,
    matched_image_path TEXT,
    loadout_snapshot TEXT,
    created_at TEXT,
    FOREIGN KEY (clip_id) REFERENCES clips(id),
    FOREIGN KEY (segment_id) REFERENCES av_segments(id)
);

CREATE TABLE IF NOT EXISTS death_screen_detections (
    id INTEGER PRIMARY KEY,
    clip_id INTEGER NOT NULL,
    segment_id INTEGER NOT NULL,
    frame_path TEXT NOT NULL,
    timestamp REAL,
    status TEXT,
    killed_with TEXT,
    killer_name TEXT,
    raw_text TEXT,
    confidence REAL DEFAULT 0,
    created_at TEXT,
    FOREIGN KEY (clip_id) REFERENCES clips(id),
    FOREIGN KEY (segment_id) REFERENCES av_segments(id)
);

CREATE TABLE IF NOT EXISTS processing_events (
    id INTEGER PRIMARY KEY,
    clip_id INTEGER,
    event_type TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    created_at TEXT,
    FOREIGN KEY (clip_id) REFERENCES clips(id)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS scan_runs (
    id INTEGER PRIMARY KEY,
    source_root TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    files_seen INTEGER DEFAULT 0,
    files_new INTEGER DEFAULT 0,
    files_changed INTEGER DEFAULT 0,
    files_unchanged INTEGER DEFAULT 0,
    files_missing INTEGER DEFAULT 0,
    status TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS clip_scan_state (
    id INTEGER PRIMARY KEY,
    clip_id INTEGER NOT NULL,
    source_root TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT,
    size_bytes INTEGER,
    modified_at TEXT,
    last_scanned_at TEXT,
    last_seen_at TEXT,
    needs_reprocess INTEGER DEFAULT 0,
    reason TEXT,
    FOREIGN KEY (clip_id) REFERENCES clips(id)
);

CREATE TABLE IF NOT EXISTS operations (
    id TEXT PRIMARY KEY,
    operation_type TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    progress_percent REAL,
    current_step TEXT,
    current_item TEXT,
    total_items INTEGER,
    completed_items INTEGER,
    failed_items INTEGER,
    skipped_items INTEGER,
    message TEXT,
    errors TEXT
);

CREATE TABLE IF NOT EXISTS operation_events (
    id INTEGER PRIMARY KEY,
    operation_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT,
    step TEXT,
    status TEXT,
    progress_percent REAL,
    current_item TEXT,
    message TEXT,
    metadata TEXT,
    FOREIGN KEY (operation_id) REFERENCES operations(id)
);

CREATE INDEX IF NOT EXISTS idx_clips_file_hash ON clips(file_hash);
CREATE INDEX IF NOT EXISTS idx_clips_filename ON clips(filename);
CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status);
CREATE INDEX IF NOT EXISTS idx_clips_scan_status ON clips(scan_status);
CREATE INDEX IF NOT EXISTS idx_clips_group_name ON clips(group_name);
CREATE INDEX IF NOT EXISTS idx_clips_relative_path ON clips(relative_path);
CREATE INDEX IF NOT EXISTS idx_av_segments_clip_id ON av_segments(clip_id);
CREATE INDEX IF NOT EXISTS idx_av_segments_group_name ON av_segments(group_name);
CREATE INDEX IF NOT EXISTS idx_av_segments_modality ON av_segments(modality);
CREATE UNIQUE INDEX IF NOT EXISTS idx_av_segments_unique_settings
    ON av_segments(clip_id, start_time, end_time, segment_settings_hash);
CREATE INDEX IF NOT EXISTS idx_transcripts_clip_id ON transcripts(clip_id);
CREATE INDEX IF NOT EXISTS idx_text_items_clip_id ON text_items(clip_id);
CREATE INDEX IF NOT EXISTS idx_hud_loadout_clip_id ON hud_loadout_detections(clip_id);
CREATE INDEX IF NOT EXISTS idx_hud_loadout_segment_id ON hud_loadout_detections(segment_id);
CREATE INDEX IF NOT EXISTS idx_hud_loadout_entity_name ON hud_loadout_detections(entity_name);
CREATE INDEX IF NOT EXISTS idx_death_screen_clip_id ON death_screen_detections(clip_id);
CREATE INDEX IF NOT EXISTS idx_death_screen_killed_with ON death_screen_detections(killed_with);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(key);
CREATE INDEX IF NOT EXISTS idx_scan_runs_source_root ON scan_runs(source_root);
CREATE INDEX IF NOT EXISTS idx_clip_scan_state_clip_id ON clip_scan_state(clip_id);
CREATE INDEX IF NOT EXISTS idx_clip_scan_state_file_path ON clip_scan_state(file_path);
CREATE UNIQUE INDEX IF NOT EXISTS idx_clip_scan_state_unique_path
    ON clip_scan_state(source_root, file_path);
CREATE INDEX IF NOT EXISTS idx_operations_status ON operations(status);
CREATE INDEX IF NOT EXISTS idx_operation_events_operation_id ON operation_events(operation_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def _representative_hud_rows(rows: list[Any]) -> list[Any]:
    if not rows:
        return []
    by_frame: dict[tuple[float, str], list[Any]] = {}
    for row in rows:
        key = (float(row["timestamp"] or 0.0), str(row["frame_path"] or ""))
        by_frame.setdefault(key, []).append(row)
    candidates: list[tuple[float, float, float, int, list[Any]]] = []
    for (timestamp, _frame_path), frame_rows in by_frame.items():
        named = [row for row in frame_rows if row["entity_name"]]
        if not named:
            continue
        current_ocr_active = any(
            int(row["is_active"] or 0)
            and str(row["slot_key"] or "") == "current_ocr"
            for row in named
        )
        active_equipment = any(int(row["is_active"] or 0) for row in named)
        weapon_count = len(_dedupe(str(row["entity_name"] or "") for row in named if row["entity_type"] == "weapon"))
        active_rank = 2.0 if current_ocr_active else 1.0 if active_equipment else 0.0
        avg_confidence = sum(float(row["confidence"] or 0.0) for row in named) / len(named)
        candidates.append((active_rank, float(weapon_count), avg_confidence, int(timestamp * 1000), frame_rows))
    if not candidates:
        return rows
    _, _, _, _, selected = max(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
    order = {"current_ocr": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "mouse5": 6, "7": 7, "8": 8, "9": 9, "0": 10}
    return sorted(selected, key=lambda row: order.get(str(row["slot_key"] or ""), 99))


def _empty_hud_summary() -> dict[str, Any]:
    return {
        "active_weapon": None,
        "active_equipment": None,
        "active_equipment_type": None,
        "loadout": [],
        "evidence": [],
    }


def _hud_evidence_payload(rows: Iterable[Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in rows:
        name = row["entity_name"]
        if not name:
            continue
        evidence.append(
            {
                "segment_id": _int_or_none(row["segment_id"]),
                "frame_path": row["frame_path"],
                "timestamp": _float_or_none(row["timestamp"]),
                "slot_key": row["slot_key"],
                "is_active": bool(int(row["is_active"] or 0)),
                "entity_id": row["entity_id"],
                "entity_name": str(name),
                "entity_type": row["entity_type"],
                "confidence": _float_or_none(row["confidence"]),
                "matched_image_path": row["matched_image_path"],
            }
        )
    return evidence


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class Database:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn = connect(self.db_path)
        init_db(self.conn)

    def close(self) -> None:
        self.conn.close()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, tuple(params))
        self.conn.commit()
        return cur

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, tuple(params)).fetchall())

    def one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, tuple(params)).fetchone()

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.one("SELECT value FROM settings WHERE key = ?", (key,))
        return str(row["value"]) if row else default

    def set_setting(self, key: str, value: str) -> None:
        self.execute(
            """
            INSERT INTO settings(key, value, updated_at) VALUES(?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, value, utc_now()),
        )

    def create_scan_run(self, source_root: str) -> int:
        cur = self.execute(
            "INSERT INTO scan_runs(source_root, started_at, status) VALUES(?, ?, ?)",
            (source_root, utc_now(), "running"),
        )
        return int(cur.lastrowid)

    def finish_scan_run(self, scan_run_id: int, stats: dict[str, Any], status: str = "completed", error: str | None = None) -> None:
        self.execute(
            """
            UPDATE scan_runs
            SET finished_at=?, files_seen=?, files_new=?, files_changed=?, files_unchanged=?,
                files_missing=?, status=?, error_message=?
            WHERE id=?
            """,
            (
                utc_now(),
                stats.get("files_seen", 0),
                stats.get("files_new", 0),
                stats.get("files_changed", 0),
                stats.get("files_unchanged", 0),
                stats.get("files_missing", 0),
                status,
                error,
                scan_run_id,
            ),
        )

    def upsert_clip(self, values: dict[str, Any]) -> int:
        now = utc_now()
        values = dict(values)
        values.setdefault("last_seen_at", now)
        values.setdefault("status", "pending")
        values.setdefault("scan_status", "unknown")

        existing = None
        if values.get("file_hash"):
            existing = self.one("SELECT id FROM clips WHERE file_hash = ?", (values["file_hash"],))
        if existing is None:
            existing = self.one(
                "SELECT id FROM clips WHERE source_root = ? AND path = ?",
                (values.get("source_root"), values.get("path")),
            )

        fields = [
            "file_hash",
            "filename",
            "path",
            "relative_path",
            "source_root",
            "group_name",
            "duration",
            "size_bytes",
            "created_at",
            "modified_at",
            "indexed_at",
            "last_seen_at",
            "width",
            "height",
            "fps",
            "codec",
            "status",
            "scan_status",
            "summary",
            "error_message",
        ]
        if existing:
            clip_id = int(existing["id"])
            assignments = ", ".join(f"{field}=?" for field in fields)
            self.execute(
                f"UPDATE clips SET {assignments} WHERE id=?",
                [values.get(field) for field in fields] + [clip_id],
            )
            return clip_id

        placeholders = ", ".join("?" for _ in fields)
        cur = self.execute(
            f"INSERT INTO clips({', '.join(fields)}) VALUES({placeholders})",
            [values.get(field) for field in fields],
        )
        return int(cur.lastrowid)

    def update_clip_status(self, clip_id: int, status: str, error_message: str | None = None, indexed: bool = False) -> None:
        indexed_at = utc_now() if indexed else self.one("SELECT indexed_at FROM clips WHERE id=?", (clip_id,))
        indexed_value = indexed_at if isinstance(indexed_at, str) else (indexed_at["indexed_at"] if indexed_at else None)
        self.execute(
            "UPDATE clips SET status=?, error_message=?, indexed_at=? WHERE id=?",
            (status, error_message, indexed_value, clip_id),
        )

    def get_clip(self, clip_id: int) -> sqlite3.Row | None:
        return self.one("SELECT * FROM clips WHERE id=?", (clip_id,))

    def list_clips(self, group_name: str | None = None, status: str | None = None) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if group_name:
            clauses.append("group_name=?")
            params.append(group_name)
        if status:
            clauses.append("status=?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.query(f"SELECT * FROM clips {where} ORDER BY group_name, filename", params)

    def upsert_scan_state(
        self,
        clip_id: int,
        source_root: str,
        file_path: str,
        file_hash: str | None,
        size_bytes: int,
        modified_at: str,
        needs_reprocess: bool,
        reason: str,
    ) -> None:
        now = utc_now()
        self.execute(
            """
            INSERT INTO clip_scan_state(
                clip_id, source_root, file_path, file_hash, size_bytes, modified_at,
                last_scanned_at, last_seen_at, needs_reprocess, reason
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_root, file_path) DO UPDATE SET
                clip_id=excluded.clip_id,
                file_hash=excluded.file_hash,
                size_bytes=excluded.size_bytes,
                modified_at=excluded.modified_at,
                last_scanned_at=excluded.last_scanned_at,
                last_seen_at=excluded.last_seen_at,
                needs_reprocess=excluded.needs_reprocess,
                reason=excluded.reason
            """,
            (clip_id, source_root, file_path, file_hash, size_bytes, modified_at, now, now, int(needs_reprocess), reason),
        )

    def get_scan_state(self, source_root: str, file_path: str) -> sqlite3.Row | None:
        return self.one(
            "SELECT * FROM clip_scan_state WHERE source_root=? AND file_path=?",
            (source_root, file_path),
        )

    def mark_missing_not_seen(self, source_root: str, seen_paths: set[str]) -> int:
        rows = self.query("SELECT clip_id, file_path FROM clip_scan_state WHERE source_root=?", (source_root,))
        count = 0
        for row in rows:
            if row["file_path"] not in seen_paths:
                count += 1
                self.execute(
                    "UPDATE clip_scan_state SET needs_reprocess=0, reason=?, last_scanned_at=? WHERE id=?",
                    ("missing", utc_now(), row["id"]),
                )
                self.execute(
                    "UPDATE clips SET scan_status=?, status=? WHERE id=?",
                    ("missing", "missing", row["clip_id"]),
                )
        return count

    def upsert_segment(self, values: dict[str, Any]) -> int:
        values = dict(values)
        values.setdefault("created_at", utc_now())
        fields = [
            "clip_id",
            "group_name",
            "start_time",
            "end_time",
            "duration",
            "modality",
            "representative_frame_path",
            "video_segment_path",
            "audio_segment_path",
            "embedding_id",
            "embedding_model",
            "embedding_precision",
            "runtime_backend",
            "segment_settings_hash",
            "created_at",
            "error_message",
        ]
        cur = self.execute(
            f"""
            INSERT INTO av_segments({', '.join(fields)}) VALUES({', '.join('?' for _ in fields)})
            ON CONFLICT(clip_id, start_time, end_time, segment_settings_hash) DO UPDATE SET
                modality=excluded.modality,
                representative_frame_path=excluded.representative_frame_path,
                video_segment_path=excluded.video_segment_path,
                audio_segment_path=excluded.audio_segment_path,
                embedding_id=COALESCE(excluded.embedding_id, av_segments.embedding_id),
                embedding_model=COALESCE(excluded.embedding_model, av_segments.embedding_model),
                embedding_precision=COALESCE(excluded.embedding_precision, av_segments.embedding_precision),
                runtime_backend=COALESCE(excluded.runtime_backend, av_segments.runtime_backend),
                error_message=excluded.error_message
            """,
            [values.get(field) for field in fields],
        )
        row = self.one(
            "SELECT id FROM av_segments WHERE clip_id=? AND start_time=? AND end_time=? AND segment_settings_hash=?",
            (values["clip_id"], values["start_time"], values["end_time"], values["segment_settings_hash"]),
        )
        return int(row["id"] if row else cur.lastrowid)

    def list_segments(self, clip_id: int | None = None) -> list[sqlite3.Row]:
        if clip_id is None:
            return self.query("SELECT * FROM av_segments ORDER BY clip_id, start_time")
        return self.query("SELECT * FROM av_segments WHERE clip_id=? ORDER BY start_time", (clip_id,))

    def add_segment_frame(self, segment_id: int, frame_path: str, timestamp: float, frame_index: int) -> None:
        existing = self.one(
            "SELECT id FROM av_segment_frames WHERE segment_id=? AND frame_index=?",
            (segment_id, frame_index),
        )
        if existing:
            self.execute(
                "UPDATE av_segment_frames SET frame_path=?, timestamp=? WHERE id=?",
                (frame_path, timestamp, existing["id"]),
            )
            return
        self.execute(
            "INSERT INTO av_segment_frames(segment_id, frame_path, timestamp, frame_index) VALUES(?, ?, ?, ?)",
            (segment_id, frame_path, timestamp, frame_index),
        )

    def add_transcript(self, values: dict[str, Any]) -> int:
        cur = self.execute(
            """
            INSERT INTO transcripts(
                clip_id, start_time, end_time, text, confidence, model_name,
                embedding_id, embedding_model, embedding_precision, runtime_backend
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["clip_id"],
                values.get("start_time"),
                values.get("end_time"),
                values["text"],
                values.get("confidence"),
                values.get("model_name"),
                values.get("embedding_id"),
                values.get("embedding_model"),
                values.get("embedding_precision"),
                values.get("runtime_backend"),
            ),
        )
        return int(cur.lastrowid)

    def replace_transcripts(self, clip_id: int, transcripts: list[dict[str, Any]]) -> None:
        self.execute("DELETE FROM transcripts WHERE clip_id=?", (clip_id,))
        for transcript in transcripts:
            transcript = dict(transcript)
            transcript["clip_id"] = clip_id
            self.add_transcript(transcript)

    def get_transcripts(self, clip_id: int) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM transcripts WHERE clip_id=? ORDER BY COALESCE(start_time, 0)", (clip_id,))

    def set_clip_tags(self, clip_id: int, tags: list[str], source: str = "rules") -> None:
        cleaned = sorted({tag.strip().lower() for tag in tags if tag.strip()})
        self.execute("DELETE FROM clip_tags WHERE clip_id=?", (clip_id,))
        for tag in cleaned:
            self.execute("INSERT OR IGNORE INTO tags(name) VALUES(?)", (tag,))
            row = self.one("SELECT id FROM tags WHERE name=?", (tag,))
            if row:
                self.execute(
                    "INSERT OR REPLACE INTO clip_tags(clip_id, tag_id, source) VALUES(?, ?, ?)",
                    (clip_id, row["id"], source),
                )

    def get_clip_tags(self, clip_id: int) -> list[str]:
        rows = self.query(
            """
            SELECT tags.name FROM tags
            JOIN clip_tags ON clip_tags.tag_id = tags.id
            WHERE clip_tags.clip_id=?
            ORDER BY tags.name
            """,
            (clip_id,),
        )
        return [str(row["name"]) for row in rows]

    def add_text_item(self, values: dict[str, Any]) -> int:
        cur = self.execute(
            """
            INSERT INTO text_items(clip_id, kind, text, embedding_id, embedding_model, embedding_precision, runtime_backend)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["clip_id"],
                values["kind"],
                values["text"],
                values.get("embedding_id"),
                values.get("embedding_model"),
                values.get("embedding_precision"),
                values.get("runtime_backend"),
            ),
        )
        return int(cur.lastrowid)

    def replace_text_items(self, clip_id: int, items: list[dict[str, Any]]) -> None:
        self.execute("DELETE FROM text_items WHERE clip_id=?", (clip_id,))
        for item in items:
            payload = dict(item)
            payload["clip_id"] = clip_id
            self.add_text_item(payload)

    def replace_hud_detections(self, clip_id: int, segment_id: int, rows: list[dict[str, Any]]) -> None:
        self.execute("DELETE FROM hud_loadout_detections WHERE clip_id=? AND segment_id=?", (clip_id, segment_id))
        for row in rows:
            payload = dict(row)
            payload["clip_id"] = clip_id
            payload["segment_id"] = segment_id
            payload.setdefault("created_at", utc_now())
            self.execute(
                """
                INSERT INTO hud_loadout_detections(
                    clip_id, segment_id, frame_path, timestamp, slot_key, is_active,
                    entity_id, entity_name, entity_type, confidence, matched_image_path,
                    loadout_snapshot, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["clip_id"],
                    payload["segment_id"],
                    payload["frame_path"],
                    payload.get("timestamp"),
                    payload["slot_key"],
                    int(payload.get("is_active") or 0),
                    payload.get("entity_id"),
                    payload.get("entity_name"),
                    payload.get("entity_type"),
                    float(payload.get("confidence") or 0.0),
                    payload.get("matched_image_path"),
                    payload.get("loadout_snapshot"),
                    payload["created_at"],
                ),
            )

    def list_hud_detections(self, clip_id: int | None = None, segment_id: int | None = None) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if clip_id is not None:
            clauses.append("clip_id=?")
            params.append(clip_id)
        if segment_id is not None:
            clauses.append("segment_id=?")
            params.append(segment_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.query(
            f"""
            SELECT * FROM hud_loadout_detections
            {where}
            ORDER BY clip_id, segment_id, frame_path, slot_key
            """,
            params,
        )

    def hud_loadout_summary(self, clip_id: int) -> dict[str, Any]:
        rows = self.list_hud_detections(clip_id=clip_id)
        death_segment_ids = {int(row["segment_id"]) for row in self.list_death_screen_detections(clip_id=clip_id)}
        if death_segment_ids:
            filtered = [row for row in rows if int(row["segment_id"]) not in death_segment_ids]
            if filtered:
                rows = filtered
        rows = _representative_hud_rows(rows)
        loadout: list[str] = []
        active_weapons: list[str] = []
        active_equipment: list[tuple[str, str | None]] = []
        weapon_slots: list[str] = []
        for row in rows:
            name = row["entity_name"]
            if not name:
                continue
            entity_type = str(row["entity_type"] or "") or None
            loadout.append(str(name))
            if entity_type == "weapon" and str(row["slot_key"] or "") in {"1", "2", "3"}:
                weapon_slots.append(str(name))
            if int(row["is_active"] or 0):
                active_equipment.append((str(name), entity_type))
                if entity_type == "weapon":
                    active_weapons.append(str(name))
        deduped_active_weapons = _dedupe(active_weapons)
        active_equipment_name = active_equipment[0][0] if active_equipment else (deduped_active_weapons[0] if deduped_active_weapons else None)
        active_equipment_type = active_equipment[0][1] if active_equipment else ("weapon" if deduped_active_weapons else None)
        current_ocr_active = bool(active_equipment) and any(str(row["slot_key"] or "") == "current_ocr" for row in rows)
        credible_weapon_loadout = len(_dedupe(weapon_slots)) >= 2
        if current_ocr_active:
            return {
                "active_weapon": deduped_active_weapons[0] if deduped_active_weapons else None,
                "active_equipment": active_equipment_name,
                "active_equipment_type": active_equipment_type,
                "loadout": _dedupe(loadout),
                "evidence": _hud_evidence_payload(rows),
            }
        if not credible_weapon_loadout and not active_weapons and not active_equipment:
            return _empty_hud_summary()
        return {
            "active_weapon": deduped_active_weapons[0] if deduped_active_weapons else None,
            "active_equipment": active_equipment_name,
            "active_equipment_type": active_equipment_type,
            "loadout": _dedupe(loadout),
            "evidence": _hud_evidence_payload(rows),
        }

    def search_hud_detections(self, terms: list[str], group_name: str | None = None) -> list[sqlite3.Row]:
        cleaned = [term.strip().lower() for term in terms if term.strip()]
        if not cleaned:
            return []
        clauses = ["h.entity_name IS NOT NULL"]
        params: list[Any] = []
        term_clauses = []
        for term in cleaned:
            term_clauses.append("LOWER(h.entity_name) LIKE ?")
            params.append(f"%{term}%")
        clauses.append("(" + " OR ".join(term_clauses) + ")")
        if group_name:
            clauses.append("c.group_name=?")
            params.append(group_name)
        return self.query(
            f"""
            SELECT h.*, c.group_name, c.filename
            FROM hud_loadout_detections h
            JOIN clips c ON c.id = h.clip_id
            WHERE {' AND '.join(clauses)}
            ORDER BY h.is_active DESC, h.confidence DESC
            """,
            params,
        )

    def search_death_screen_detections(self, terms: list[str], group_name: str | None = None) -> list[sqlite3.Row]:
        cleaned = [term.strip().lower() for term in terms if term.strip()]
        if not cleaned:
            return []
        clauses = ["d.killed_with IS NOT NULL"]
        params: list[Any] = []
        term_clauses = []
        for term in cleaned:
            term_clauses.append("(LOWER(d.killed_with) LIKE ? OR LOWER(d.killer_name) LIKE ?)")
            params.extend([f"%{term}%", f"%{term}%"])
        clauses.append("(" + " OR ".join(term_clauses) + ")")
        if group_name:
            clauses.append("c.group_name=?")
            params.append(group_name)
        return self.query(
            f"""
            SELECT d.*, c.group_name, c.filename
            FROM death_screen_detections d
            JOIN clips c ON c.id = d.clip_id
            WHERE {' AND '.join(clauses)}
            ORDER BY d.confidence DESC, d.timestamp DESC
            """,
            params,
        )

    def replace_death_screen_detection(self, clip_id: int, segment_id: int, row: dict[str, Any] | None) -> None:
        self.execute("DELETE FROM death_screen_detections WHERE clip_id=? AND segment_id=?", (clip_id, segment_id))
        if row is None:
            return
        payload = dict(row)
        payload["clip_id"] = clip_id
        payload["segment_id"] = segment_id
        payload.setdefault("created_at", utc_now())
        self.execute(
            """
            INSERT INTO death_screen_detections(
                clip_id, segment_id, frame_path, timestamp, status, killed_with,
                killer_name, raw_text, confidence, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["clip_id"],
                payload["segment_id"],
                payload["frame_path"],
                payload.get("timestamp"),
                payload.get("status"),
                payload.get("killed_with"),
                payload.get("killer_name"),
                payload.get("raw_text"),
                float(payload.get("confidence") or 0.0),
                payload["created_at"],
            ),
        )

    def list_death_screen_detections(self, clip_id: int | None = None) -> list[sqlite3.Row]:
        params: list[Any] = []
        where = ""
        if clip_id is not None:
            where = "WHERE clip_id=?"
            params.append(clip_id)
        return self.query(
            f"""
            SELECT * FROM death_screen_detections
            {where}
            ORDER BY clip_id, timestamp DESC, id DESC
            """,
            params,
        )

    def death_screen_summary(self, clip_id: int) -> dict[str, Any]:
        rows = self.list_death_screen_detections(clip_id=clip_id)
        best = next((row for row in rows if row["killed_with"]), rows[0] if rows else None)
        if best is None:
            return {"status": None, "killed_with": None, "killer_name": None}
        return {
            "status": best["status"],
            "killed_with": best["killed_with"],
            "killer_name": best["killer_name"],
            "confidence": best["confidence"],
            "frame_path": best["frame_path"],
            "timestamp": best["timestamp"],
            "raw_text": best["raw_text"],
        }

    def add_processing_event(
        self,
        clip_id: int | None,
        event_type: str,
        status: str,
        message: str | None = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO processing_events(clip_id, event_type, status, message, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (clip_id, event_type, status, message, utc_now()),
        )

    def groups(self) -> list[sqlite3.Row]:
        return self.query(
            """
            SELECT
                group_name,
                COUNT(*) AS total_videos,
                SUM(CASE WHEN status='indexed' THEN 1 ELSE 0 END) AS indexed_videos,
                SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed_videos,
                SUM(CASE WHEN status='missing' THEN 1 ELSE 0 END) AS missing_videos,
                MAX(indexed_at) AS last_indexed_at
            FROM clips
            GROUP BY group_name
            ORDER BY group_name
            """
        )

    def create_operation(self, operation_id: str, operation_type: str) -> None:
        self.execute(
            """
            INSERT INTO operations(
                id, operation_type, status, created_at, progress_percent, total_items,
                completed_items, failed_items, skipped_items
            ) VALUES(?, ?, ?, ?, 0, 0, 0, 0, 0)
            """,
            (operation_id, operation_type, "queued", utc_now()),
        )

    def update_operation(self, operation_id: str, **updates: Any) -> None:
        if not updates:
            return
        fields = ", ".join(f"{key}=?" for key in updates)
        self.execute(f"UPDATE operations SET {fields} WHERE id=?", [*updates.values(), operation_id])

    def get_operation(self, operation_id: str) -> OperationRecord | None:
        row = self.one("SELECT * FROM operations WHERE id=?", (operation_id,))
        return OperationRecord(**dict(row)) if row else None

    def list_operations(self) -> list[OperationRecord]:
        rows = self.query("SELECT * FROM operations ORDER BY COALESCE(started_at, created_at) DESC")
        return [OperationRecord(**dict(row)) for row in rows]

    def add_operation_event(self, event: OperationEvent) -> None:
        self.execute(
            """
            INSERT INTO operation_events(
                operation_id, timestamp, event_type, step, status, progress_percent,
                current_item, message, metadata
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.operation_id,
                event.timestamp,
                event.event_type,
                event.step,
                event.status,
                event.progress_percent,
                event.current_item,
                event.message,
                json.dumps(event.metadata),
            ),
        )

    def list_operation_events(self, operation_id: str, after_id: int | None = None) -> list[OperationEvent]:
        if after_id is None:
            rows = self.query("SELECT * FROM operation_events WHERE operation_id=? ORDER BY id", (operation_id,))
        else:
            rows = self.query(
                "SELECT * FROM operation_events WHERE operation_id=? AND id>? ORDER BY id",
                (operation_id, after_id),
            )
        events: list[OperationEvent] = []
        for row in rows:
            data = dict(row)
            data["metadata"] = json.loads(data.get("metadata") or "{}")
            events.append(OperationEvent(**data))
        return events

    def reset_index(self) -> None:
        for table in [
            "operation_events",
            "operations",
            "processing_events",
            "text_items",
            "clip_tags",
            "tags",
            "transcripts",
            "audio_artifacts",
            "av_segment_frames",
            "av_segments",
            "clip_scan_state",
            "scan_runs",
            "clips",
        ]:
            self.execute(f"DELETE FROM {table}")
