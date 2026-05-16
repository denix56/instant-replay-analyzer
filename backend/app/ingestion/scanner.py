from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

from ..config import SUPPORTED_VIDEO_EXTENSIONS
from ..db import Database
from ..models import ScanSummary, utc_now
from .video_metadata import read_video_metadata


VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
SIDECAR_EXTENSIONS = {".aac", ".json", ".m4a", ".srt", ".txt", ".vtt", ".wav"}
DEFAULT_SCAN_EXTENSIONS = VIDEO_EXTENSIONS | SIDECAR_EXTENSIONS
HashPolicy = Literal["never", "always", "duplicates"]


@dataclass(frozen=True)
class DiscoveredFile:
    path: Path
    size_bytes: int
    mtime_ns: int
    suffix: str
    sha256: str | None = None

    @property
    def is_video(self) -> bool:
        return self.suffix in VIDEO_EXTENSIONS


@dataclass(frozen=True)
class ClipGroup:
    key: str
    primary_video: Path
    related_files: tuple[Path, ...]


@dataclass(frozen=True)
class ScanResult:
    root: Path
    files: tuple[DiscoveredFile, ...]
    groups: tuple[ClipGroup, ...]


def normalize_extension(extension: str) -> str:
    extension = extension.lower().strip()
    if not extension:
        raise ValueError("extension cannot be empty")
    return extension if extension.startswith(".") else f".{extension}"


def normalize_clip_key(path: Path) -> str:
    """Return a stable grouping key for gameplay clips and sidecar artifacts."""
    return path.stem.casefold()


def iter_candidate_files(
    root: str | Path,
    extensions: Iterable[str] = DEFAULT_SCAN_EXTENSIONS,
    *,
    include_hidden: bool = False,
) -> Iterator[Path]:
    root_path = Path(root).expanduser().resolve()
    wanted = {normalize_extension(ext) for ext in extensions}

    if not root_path.exists():
        return
    if root_path.is_file():
        if root_path.suffix.lower() in wanted:
            yield root_path
        return

    paths: list[Path] = []
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if not include_hidden and any(part.startswith(".") for part in path.relative_to(root_path).parts):
            continue
        if path.suffix.lower() in wanted:
            paths.append(path.resolve())

    yield from sorted(paths, key=lambda item: item.as_posix().casefold())


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_files(
    root: str | Path,
    *,
    extensions: Iterable[str] = DEFAULT_SCAN_EXTENSIONS,
    hash_policy: HashPolicy = "duplicates",
) -> tuple[DiscoveredFile, ...]:
    paths = tuple(iter_candidate_files(root, extensions))
    stats = {path: path.stat() for path in paths}

    duplicate_keys: set[tuple[str, int]] = set()
    if hash_policy == "duplicates":
        seen: dict[tuple[str, int], int] = {}
        for path, stat in stats.items():
            key = (path.name.casefold(), stat.st_size)
            seen[key] = seen.get(key, 0) + 1
        duplicate_keys = {key for key, count in seen.items() if count > 1}

    discovered: list[DiscoveredFile] = []
    for path in paths:
        stat = stats[path]
        duplicate_key = (path.name.casefold(), stat.st_size)
        should_hash = hash_policy == "always" or (
            hash_policy == "duplicates" and duplicate_key in duplicate_keys
        )
        discovered.append(
            DiscoveredFile(
                path=path,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                suffix=path.suffix.lower(),
                sha256=sha256_file(path) if should_hash else None,
            )
        )
    return tuple(discovered)


def detect_clip_groups(files: Sequence[DiscoveredFile | Path]) -> tuple[ClipGroup, ...]:
    by_key: dict[str, list[Path]] = {}
    for item in files:
        path = item.path if isinstance(item, DiscoveredFile) else Path(item)
        by_key.setdefault(normalize_clip_key(path), []).append(path)

    groups: list[ClipGroup] = []
    for key in sorted(by_key):
        paths = sorted(by_key[key], key=lambda item: item.as_posix().casefold())
        videos = [path for path in paths if path.suffix.lower() in VIDEO_EXTENSIONS]
        if not videos:
            continue
        primary = videos[0]
        related = tuple(path for path in paths if path != primary)
        groups.append(ClipGroup(key=key, primary_video=primary, related_files=related))

    return tuple(groups)


def scan_library(
    root: str | Path,
    *,
    extensions: Iterable[str] = DEFAULT_SCAN_EXTENSIONS,
    hash_policy: HashPolicy = "duplicates",
) -> ScanResult:
    root_path = Path(root).expanduser().resolve()
    files = discover_files(root_path, extensions=extensions, hash_policy=hash_policy)
    return ScanResult(root=root_path, files=files, groups=detect_clip_groups(files))


def iter_supported_videos(root: str | Path) -> Iterator[Path]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
        return
    for path in sorted(root_path.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            yield path.resolve()


def group_name_for_path(root: str | Path, path: str | Path) -> str:
    root_path = Path(root).expanduser().resolve()
    video_path = Path(path).expanduser().resolve()
    try:
        parts = video_path.relative_to(root_path).parts
    except ValueError:
        return "Ungrouped"
    if len(parts) > 1:
        return parts[0]
    return "Ungrouped"


def relative_video_path(root: str | Path, path: str | Path) -> str:
    try:
        return str(Path(path).expanduser().resolve().relative_to(Path(root).expanduser().resolve()))
    except ValueError:
        return Path(path).name


def scan_directory(
    root: str | Path,
    db: Database,
    *,
    force_verify: bool = False,
    progress_callback: object | None = None,
) -> ScanSummary:
    """Scan supported videos, persist scan state, and mark missing clips.

    The fast path compares path, size, and mtime. Hashes are computed only for
    new/changed files, missing hash state, or explicit verification.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"invalid input folder: {root_path}")

    scan_run_id = db.create_scan_run(str(root_path))
    summary = ScanSummary(source_root=str(root_path))
    seen_paths: set[str] = set()

    try:
        all_files = [path for path in root_path.rglob("*") if path.is_file()]
        unsupported = [path for path in all_files if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS]
        videos = [path.resolve() for path in all_files if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS]
        summary.unsupported_files = len(unsupported)
        summary.supported_videos = len(videos)

        for index, video_path in enumerate(sorted(videos, key=lambda item: item.as_posix().casefold()), start=1):
            if progress_callback:
                _emit_progress(
                    progress_callback,
                    message=f"Scanning {video_path.name}",
                    progress=index / max(len(videos), 1),
                    data={"current_folder": str(video_path.parent), "supported_videos": len(videos)},
                )
            stat = video_path.stat()
            full_path = str(video_path)
            seen_paths.add(full_path)
            modified_marker = str(stat.st_mtime_ns)
            created_at = _timestamp(stat.st_ctime)
            modified_at = _timestamp(stat.st_mtime)
            group_name = group_name_for_path(root_path, video_path)
            summary.groups[group_name] = summary.groups.get(group_name, 0) + 1

            state = db.get_scan_state(str(root_path), full_path)
            unchanged = (
                state is not None
                and int(state["size_bytes"] or -1) == stat.st_size
                and str(state["modified_at"]) == modified_marker
                and not force_verify
            )

            if unchanged:
                summary.files_unchanged += 1
                clip_id = int(state["clip_id"])
                db.execute(
                    "UPDATE clips SET last_seen_at=?, scan_status=? WHERE id=?",
                    (utc_now(), "unchanged", clip_id),
                )
                db.upsert_scan_state(
                    clip_id,
                    str(root_path),
                    full_path,
                    state["file_hash"],
                    stat.st_size,
                    modified_marker,
                    False,
                    "unchanged",
                )
                continue

            file_hash = sha256_file(video_path)
            reason = "new" if state is None else "changed"
            if state is None:
                summary.files_new += 1
            else:
                summary.files_changed += 1

            metadata = read_video_metadata(video_path)
            existing_status = "pending"
            if state is not None:
                row = db.get_clip(int(state["clip_id"]))
                existing_status = str(row["status"]) if row else "pending"
            clip_id = db.upsert_clip(
                {
                    "file_hash": file_hash,
                    "filename": video_path.name,
                    "path": full_path,
                    "relative_path": relative_video_path(root_path, video_path),
                    "source_root": str(root_path),
                    "group_name": group_name,
                    "duration": metadata.duration_seconds,
                    "size_bytes": stat.st_size,
                    "created_at": created_at,
                    "modified_at": modified_at,
                    "last_seen_at": utc_now(),
                    "width": metadata.width,
                    "height": metadata.height,
                    "fps": metadata.fps,
                    "codec": metadata.video_codec,
                    "status": "pending" if reason == "new" or existing_status == "missing" else existing_status,
                    "scan_status": reason,
                    "error_message": metadata.error,
                }
            )
            db.upsert_scan_state(
                clip_id,
                str(root_path),
                full_path,
                file_hash,
                stat.st_size,
                modified_marker,
                True,
                reason,
            )

        summary.files_seen = len(videos)
        summary.files_missing = db.mark_missing_not_seen(str(root_path), seen_paths)
        db.finish_scan_run(scan_run_id, summary.model_dump(), status="completed")
        return summary
    except Exception as exc:
        summary.status = "failed"
        summary.error_message = str(exc)
        db.finish_scan_run(scan_run_id, summary.model_dump(), status="failed", error=str(exc))
        raise


def _timestamp(seconds: float) -> str:
    from datetime import datetime

    return datetime.utcfromtimestamp(seconds).replace(microsecond=0).isoformat() + "Z"


def _emit_progress(callback: object, **kwargs: object) -> None:
    if callable(callback):
        callback(**kwargs)
