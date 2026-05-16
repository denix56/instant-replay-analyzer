from __future__ import annotations

import asyncio
import dataclasses
import importlib
import inspect
import threading
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .events import EventBus
from .schemas import EventType, OperationState, OperationStatus, ProgressEvent, utc_now

PipelineCallable = Callable[..., Any]


class OperationNotFound(KeyError):
    pass


class OperationCancelled(Exception):
    pass


@dataclass
class OperationRecord:
    status: OperationStatus
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task[None]] = None


class OperationManager:
    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.events = event_bus or EventBus()
        self._operations: Dict[str, OperationRecord] = {}
        self._lock = asyncio.Lock()

    async def start(self, kind: str, params: Optional[Dict[str, Any]] = None) -> OperationStatus:
        operation_id = uuid.uuid4().hex
        now = utc_now()
        status = OperationStatus(
            operation_id=operation_id,
            kind=kind,
            state=OperationState.PENDING,
            params=params or {},
            created_at=now,
            updated_at=now,
        )
        record = OperationRecord(status=status)
        async with self._lock:
            self._operations[operation_id] = record
        await self._publish(record, EventType.SNAPSHOT, "operation queued")
        record.task = asyncio.create_task(self._run(record), name=f"operation:{kind}:{operation_id}")
        return record.status

    async def get(self, operation_id: str) -> OperationStatus:
        return (await self._record(operation_id)).status

    async def list(self) -> List[OperationStatus]:
        async with self._lock:
            return [record.status for record in self._operations.values()]

    async def cancel(self, operation_id: str) -> OperationStatus:
        record = await self._record(operation_id)
        if record.status.state in {OperationState.SUCCEEDED, OperationState.FAILED, OperationState.CANCELED}:
            return record.status

        record.status.cancel_requested = True
        record.status.updated_at = utc_now()
        record.cancel_event.set()
        if record.task and not record.task.done():
            record.task.cancel()
        await self._publish(record, EventType.CANCELED, "cancellation requested")
        return record.status

    async def wait(self, operation_id: str) -> OperationStatus:
        record = await self._record(operation_id)
        if record.task is not None:
            try:
                await record.task
            except asyncio.CancelledError:
                pass
        return record.status

    async def _record(self, operation_id: str) -> OperationRecord:
        async with self._lock:
            record = self._operations.get(operation_id)
        if record is None:
            raise OperationNotFound(operation_id)
        return record

    async def _run(self, record: OperationRecord) -> None:
        try:
            record.status.state = OperationState.RUNNING
            record.status.started_at = utc_now()
            record.status.updated_at = record.status.started_at
            await self._publish(record, EventType.PROGRESS, "operation started")
            await asyncio.sleep(0.05)

            result = await self._call_pipeline(record)

            if record.cancel_event.is_set():
                raise OperationCancelled()

            record.status.state = OperationState.SUCCEEDED
            record.status.progress = 1.0
            record.status.message = "operation completed"
            record.status.result = result
            record.status.finished_at = utc_now()
            record.status.updated_at = record.status.finished_at
            await self._publish(record, EventType.COMPLETED, "operation completed", data=result)
        except (asyncio.CancelledError, OperationCancelled):
            record.status.state = OperationState.CANCELED
            record.status.message = "operation canceled"
            record.status.cancel_requested = True
            record.status.finished_at = utc_now()
            record.status.updated_at = record.status.finished_at
            await self._publish(record, EventType.CANCELED, "operation canceled")
        except Exception as exc:  # noqa: BLE001 - operation boundaries should capture pipeline failures.
            record.status.state = OperationState.FAILED
            record.status.error = str(exc)
            record.status.message = "operation failed"
            record.status.finished_at = utc_now()
            record.status.updated_at = record.status.finished_at
            await self._publish(record, EventType.FAILED, "operation failed", error=str(exc))

    async def _set_progress(
        self,
        record: OperationRecord,
        *,
        message: Optional[str] = None,
        progress: Optional[float] = None,
        data: Any = None,
    ) -> None:
        if progress is not None:
            record.status.progress = max(0.0, min(1.0, float(progress)))
        if message is not None:
            record.status.message = message
        record.status.updated_at = utc_now()
        await self._publish(record, EventType.PROGRESS, record.status.message, data=data)

    async def _publish(
        self,
        record: OperationRecord,
        event_type: EventType,
        message: str,
        *,
        data: Any = None,
        error: Optional[str] = None,
    ) -> None:
        await self.events.publish(
            ProgressEvent(
                sequence=0,
                type=event_type,
                operation_id=record.status.operation_id,
                state=record.status.state,
                progress=record.status.progress,
                message=message,
                data=data,
                error=error,
            )
        )

    async def _call_pipeline(self, record: OperationRecord) -> Any:
        function = self._resolve_pipeline(record.status.kind)
        if function is None:
            return await self._fallback_pipeline(record)

        loop = asyncio.get_running_loop()

        def progress_callback(*args: Any, **kwargs: Any) -> None:
            message = kwargs.pop("message", None)
            progress = kwargs.pop("progress", None)
            data = kwargs.pop("data", None)
            if args:
                if isinstance(args[0], (int, float)):
                    progress = args[0]
                    if len(args) > 1:
                        message = args[1]
                else:
                    message = args[0]
                    if len(args) > 1 and isinstance(args[1], (int, float)):
                        progress = args[1]
            asyncio.run_coroutine_threadsafe(
                self._set_progress(record, message=message, progress=progress, data=data or kwargs or None),
                loop,
            )

        call_kwargs = self._build_call_kwargs(function, record, progress_callback)

        if inspect.iscoroutinefunction(function):
            return await function(**call_kwargs)
        return await asyncio.to_thread(function, **call_kwargs)

    def _build_call_kwargs(
        self,
        function: PipelineCallable,
        record: OperationRecord,
        progress_callback: Callable[..., None],
    ) -> Dict[str, Any]:
        signature = inspect.signature(function)
        params = signature.parameters
        accepts_var_kwargs = any(item.kind == inspect.Parameter.VAR_KEYWORD for item in params.values())
        kwargs: Dict[str, Any] = {}

        for name in ("params", "payload", "request"):
            if name in params:
                kwargs[name] = record.status.params
                break

        for name in ("progress_callback", "progress_cb", "on_progress", "progress"):
            if name in params:
                kwargs[name] = progress_callback
                break

        for name in ("cancel_event", "cancellation_event"):
            if name in params:
                kwargs[name] = record.cancel_event
                break

        for name in ("should_cancel", "cancel_requested"):
            if name in params:
                kwargs[name] = record.cancel_event.is_set
                break

        for name, value in record.status.params.items():
            if accepts_var_kwargs or name in params:
                kwargs.setdefault(name, value)

        return kwargs

    def _resolve_pipeline(self, kind: str) -> Optional[PipelineCallable]:
        candidates = {
            "scan": (
                ("app.pipeline", ("run_scan", "scan_clips", "scan")),
                ("backend.app.pipeline", ("run_scan", "scan_clips", "scan")),
            ),
            "index": (
                ("app.pipeline", ("run_indexing", "index_clips", "ingest_clips")),
                ("app.pipeline.indexing", ("run", "index_clips", "index_replays")),
                ("app.indexing", ("run", "index_clips", "index_replays")),
                ("backend.app.pipeline", ("run_indexing", "index_clips", "ingest_clips")),
                ("backend.app.pipeline.indexing", ("run", "index_clips", "index_replays")),
                ("backend.app.indexing", ("run", "index_clips", "index_replays")),
            ),
            "search": (
                ("app.pipeline", ("run_search", "search_clips", "semantic_search")),
                ("app.pipeline.search", ("run", "search_clips", "semantic_search")),
                ("app.search", ("semantic_search", "search_clips")),
                ("backend.app.pipeline", ("run_search", "search_clips", "semantic_search")),
                ("backend.app.pipeline.search", ("run", "search_clips", "semantic_search")),
                ("backend.app.search", ("semantic_search", "search_clips")),
            ),
            "analyze": (
                ("app.pipeline", ("run_analysis", "analyze_clip", "analyze")),
                ("app.pipeline.analysis", ("run", "analyze_clip", "analyze")),
                ("app.analysis", ("analyze_clip", "analyze")),
                ("backend.app.pipeline", ("run_analysis", "analyze_clip", "analyze")),
                ("backend.app.pipeline.analysis", ("run", "analyze_clip", "analyze")),
                ("backend.app.analysis", ("analyze_clip", "analyze")),
            ),
        }.get(kind, ())

        for module_name, attribute_names in candidates:
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            for attribute_name in attribute_names:
                function = getattr(module, attribute_name, None)
                if callable(function):
                    return function
        return None

    async def _fallback_pipeline(self, record: OperationRecord) -> Any:
        steps: Iterable[Tuple[float, str]] = (
            (0.2, "preparing inputs"),
            (0.45, "extracting gameplay signals"),
            (0.7, "updating semantic index"),
            (0.9, "finalizing results"),
        )
        for progress, message in steps:
            if record.cancel_event.is_set():
                raise OperationCancelled()
            await asyncio.sleep(0.05)
            await self._set_progress(record, progress=progress, message=message)

        if record.status.kind == "scan":
            return self._scan_with_adjacent_scanner(record)

        if record.status.kind == "search":
            return self._search_with_adjacent_service(record)

        if record.status.kind == "index":
            indexed = self._index_with_adjacent_scanner(record)
            if indexed is not None:
                return indexed

        return {"kind": record.status.kind, "fallback": True, "thread": threading.current_thread().name}

    def _scan_with_adjacent_scanner(self, record: OperationRecord) -> Optional[Dict[str, Any]]:
        root = record.status.params.get("input") or record.status.params.get("source") or record.status.params.get("replay_dir")
        if not root:
            return None
        try:
            from ..ingestion.scanner import scan_library
        except ModuleNotFoundError:
            return None
        result = scan_library(root)
        payload = _to_plain(result)
        return {
            "kind": "scan",
            "fallback": False,
            "pipeline": "app.ingestion.scanner.scan_library",
            "scan": payload,
            "file_count": len(payload.get("files", ())),
            "clip_group_count": len(payload.get("groups", ())),
        }

    def _index_with_adjacent_scanner(self, record: OperationRecord) -> Optional[Dict[str, Any]]:
        root = record.status.params.get("source") or record.status.params.get("replay_dir")
        if not root:
            return None
        try:
            from ..ingestion.scanner import scan_library
        except ModuleNotFoundError:
            return None

        result = scan_library(root)
        payload = _to_plain(result)
        return {
            "kind": "index",
            "fallback": False,
            "pipeline": "app.ingestion.scanner.scan_library",
            "scan": payload,
            "file_count": len(payload.get("files", ())),
            "clip_group_count": len(payload.get("groups", ())),
        }

    def _search_with_adjacent_service(self, record: OperationRecord) -> Dict[str, Any]:
        query = str(record.status.params.get("query", ""))
        limit = int(record.status.params.get("limit", 10))
        filters = dict(record.status.params.get("filters") or {})
        try:
            from ..llm.schemas import SearchRequest
            from ..search.query import ClipSearchService
        except ModuleNotFoundError:
            return {"query": query, "clips": [], "limit": limit, "fallback": True}

        request = SearchRequest(
            query=query,
            limit=limit,
            game=filters.get("game"),
            tags=list(filters.get("tags") or []),
        )
        results = ClipSearchService().search(request)
        return {
            "query": query,
            "clips": _to_plain(results),
            "limit": limit,
            "fallback": False,
            "pipeline": "app.search.query.ClipSearchService.search",
        }


def _to_plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: _to_plain(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    return value
