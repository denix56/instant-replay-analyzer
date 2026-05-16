# Data Model

SQLite stores durable metadata and operation state. Qdrant stores dense vectors.

Core tables:

- `clips`: one row per source video, including path, group, media metadata, scan status, processing status, summary, and error state.
- `clip_scan_state`: path/size/mtime/hash state used to skip unchanged files.
- `scan_runs`: aggregate scan statistics.
- `av_segments`: overlapping time ranges and cached artifact paths.
- `av_segment_frames`: representative frames for each segment.
- `audio_artifacts`: extracted clip-level or segment-level audio state.
- `transcripts`: ASR text with timestamps and embedding metadata.
- `tags` and `clip_tags`: normalized tag storage.
- `text_items`: embedded summary/tag/filename/group/metadata text.
- `processing_events`: clip-level processing log.
- `settings`: persisted user settings such as clips folder and preset.
- `operations` and `operation_events`: long-running scan/index/search/analysis progress.

Important indexes cover file hash, filename, group, status, scan status, relative path, segment clip/group/modality, transcript clip, tags, scan run root, scan state path, operation status, and operation event operation ID.

Idempotency is enforced by checking scan state before processing and by unique segment settings per clip/time range.
