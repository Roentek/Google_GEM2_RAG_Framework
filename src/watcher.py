"""Watches the data/ folder and keeps Supabase embeddings in sync.

startup_sync      — diffs data/ against the DB on startup.
watch_data_folder — live FS-event loop.
sync_then_watch   — runs both sequentially in one asyncio task.

All three accept an optional on_event(type, message) callback that the
server uses to push notifications to connected browser clients via SSE.
"""

import asyncio
import logging
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from watchfiles import awatch, Change

from src.config import Config
from src.gemini_embedder import GeminiEmbedder
from src.supabase_client import SupabaseVectorClient

log = logging.getLogger("watcher")

# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------
_TEXT_EXT  = {".txt", ".md", ".pdf", ".docx", ".doc"}
_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg"}
_ALL_EXT   = _TEXT_EXT | _IMAGE_EXT | _VIDEO_EXT | _AUDIO_EXT

_IGNORE_NAMES = {".gitkeep", ".DS_Store", "Thumbs.db"}
_SUFFIX_RE    = re.compile(r"::(chunk_\d+|frame_\d+s?)$")

OnEvent = Callable[[str, str], None] | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _should_ignore(path: Path) -> bool:
    return path.name in _IGNORE_NAMES or path.name.startswith(".")


def _base_path(source: str) -> str:
    return _SUFFIX_RE.sub("", source)


def _file_mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _parse_db_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _emit(on_event: OnEvent, event_type: str, message: str) -> None:
    if on_event:
        try:
            on_event(event_type, message)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shared ingest helper
# ---------------------------------------------------------------------------
async def _ingest(
    path: Path,
    embedder: GeminiEmbedder,
    db: SupabaseVectorClient,
    config: Config,
    on_event: OnEvent = None,
) -> None:
    ext = path.suffix.lower()

    if ext in _TEXT_EXT:
        from src.ingestors.text_ingestor import TextIngestor
        result = await TextIngestor(embedder, db, config).ingest(str(path))
    elif ext in _IMAGE_EXT:
        from src.ingestors.image_ingestor import ImageIngestor
        result = await ImageIngestor(embedder, db, config).ingest(str(path))
    elif ext in _VIDEO_EXT:
        from src.ingestors.video_ingestor import VideoIngestor
        result = await VideoIngestor(embedder, db, config).ingest(str(path))
    elif ext in _AUDIO_EXT:
        from src.ingestors.audio_ingestor import AudioIngestor
        result = await AudioIngestor(embedder, db, config).ingest(str(path))
    else:
        return

    if result.errors:
        msg = f"{path.name} — {result.errors[0]}"
        log.warning("watcher: ingested %s with errors: %s", path.name, result.errors)
        _emit(on_event, "error", msg)
    else:
        msg = f"{path.name} ({result.chunks_ingested} chunk{'s' if result.chunks_ingested != 1 else ''})"
        log.info("watcher: ingested %s", msg)
        _emit(on_event, "ingested", msg)


# ---------------------------------------------------------------------------
# Startup sync
# ---------------------------------------------------------------------------
async def startup_sync(
    embedder: GeminiEmbedder,
    db: SupabaseVectorClient,
    config: Config,
    data_dir: Path,
    on_event: OnEvent = None,
) -> None:
    log.info("startup-sync: scanning %s …", data_dir.resolve())
    _emit(on_event, "sync_start", "Scanning data folder…")

    # 1. Collect all supported files on disk
    fs_files: dict[str, datetime] = {}
    for p in data_dir.rglob("*"):
        if p.is_file() and not _should_ignore(p) and p.suffix.lower() in _ALL_EXT:
            fs_files[str(p)] = _file_mtime_utc(p)

    # 2. Query DB for all sources, group by base file path → max updated_at
    db_rows = db.get_all_sources()
    db_files: dict[str, datetime | None] = {}
    for row in db_rows:
        base = _base_path(row["source"])
        ts   = _parse_db_ts(row.get("updated_at"))
        if base not in db_files:
            db_files[base] = ts
        elif ts is not None and (db_files[base] is None or ts > db_files[base]):
            db_files[base] = ts

    # 3. Classify
    to_add:    list[str] = []
    to_update: list[str] = []
    to_delete: list[str] = []

    for file_path, mtime in fs_files.items():
        if file_path not in db_files:
            to_add.append(file_path)
        else:
            db_ts = db_files[file_path]
            if db_ts is None or mtime > db_ts:
                to_update.append(file_path)

    for db_path in db_files:
        if db_path not in fs_files:
            to_delete.append(db_path)

    summary = f"{len(to_add)} new · {len(to_update)} modified · {len(to_delete)} removed"
    log.info("startup-sync: %s (disk: %d, db: %d)", summary, len(fs_files), len(db_files))
    _emit(on_event, "sync_scan", summary)

    if not (to_add or to_update or to_delete):
        log.info("startup-sync: data folder is up to date.")
        _emit(on_event, "sync_complete", "Data folder is up to date.")
        return

    # 4. Delete stale
    for path_str in to_delete:
        try:
            db.delete_by_source_prefix(path_str)
            name = Path(path_str).name
            log.info("startup-sync: removed embeddings for %s", name)
            _emit(on_event, "removed", name)
        except Exception:
            log.exception("startup-sync: failed to delete %s", path_str)

    # 5. Purge old chunks/frames for modified files
    for path_str in to_update:
        try:
            db.delete_by_source_prefix(path_str)
        except Exception:
            log.exception("startup-sync: failed to purge stale embeddings for %s", path_str)

    # 6. Ingest new + modified
    for path_str in to_add + to_update:
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            await _ingest(path, embedder, db, config, on_event)
        except Exception:
            log.exception("startup-sync: failed to ingest %s", path_str)

    log.info("startup-sync: complete.")
    _emit(on_event, "sync_complete", f"Sync complete — {summary}")


# ---------------------------------------------------------------------------
# Live watcher
# ---------------------------------------------------------------------------
async def watch_data_folder(
    embedder: GeminiEmbedder,
    db: SupabaseVectorClient,
    config: Config,
    data_dir: Path,
    on_event: OnEvent = None,
) -> None:
    log.info("watcher: live-watching %s", data_dir.resolve())
    _emit(on_event, "watching", "Live watching data folder for changes")

    pending: dict[str, Change] = {}
    DEBOUNCE_S = 1.5

    async for raw_changes in awatch(data_dir):
        for change_type, path_str in raw_changes:
            path = Path(path_str)
            if _should_ignore(path) or path.suffix.lower() not in _ALL_EXT:
                continue
            existing = pending.get(path_str)
            if change_type == Change.deleted or existing is None:
                pending[path_str] = change_type
            elif existing != Change.deleted:
                pending[path_str] = change_type

        if not pending:
            continue

        await asyncio.sleep(DEBOUNCE_S)

        for path_str, change_type in list(pending.items()):
            pending.pop(path_str)
            path = Path(path_str)
            try:
                if change_type == Change.deleted:
                    db.delete_by_source_prefix(path_str)
                    name = path.name
                    log.info("watcher: removed embeddings for %s", name)
                    _emit(on_event, "removed", name)
                else:
                    db.delete_by_source_prefix(path_str)
                    await _ingest(path, embedder, db, config, on_event)
            except Exception:
                log.exception("watcher: error processing %s", path_str)
                _emit(on_event, "error", f"Error processing {path.name}")


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------
async def sync_then_watch(
    embedder: GeminiEmbedder,
    db: SupabaseVectorClient,
    config: Config,
    data_dir: Path,
    on_event: OnEvent = None,
) -> None:
    await startup_sync(embedder, db, config, data_dir, on_event)
    await watch_data_folder(embedder, db, config, data_dir, on_event)
