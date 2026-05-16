from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from datetime import datetime
from typing import AsyncIterator, Deque, Dict, Iterable, Optional, Set

from .schemas import EventType, OperationState, ProgressEvent


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def model_dump(model: object) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")  # type: ignore[attr-defined]
    return model.dict()  # type: ignore[attr-defined]


def sse_format(event: ProgressEvent) -> str:
    payload = json.dumps(model_dump(event), default=_json_default, separators=(",", ":"))
    return f"id: {event.sequence}\nevent: {event.type.value}\ndata: {payload}\n\n"


class EventBus:
    def __init__(self, history_size: int = 100, queue_size: int = 200) -> None:
        self._history_size = history_size
        self._queue_size = queue_size
        self._history: Dict[str, Deque[ProgressEvent]] = defaultdict(lambda: deque(maxlen=history_size))
        self._subscribers: Dict[str, Set[asyncio.Queue[ProgressEvent]]] = defaultdict(set)
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def publish(self, event: ProgressEvent) -> ProgressEvent:
        async with self._lock:
            self._sequence += 1
            event.sequence = self._sequence
            self._history[event.operation_id].append(event)
            subscribers = list(self._subscribers.get(event.operation_id, set()))

        for queue in subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)
        return event

    def history(self, operation_id: str) -> Iterable[ProgressEvent]:
        return tuple(self._history.get(operation_id, ()))

    async def subscribe(
        self,
        operation_id: str,
        *,
        replay: bool = True,
        heartbeat_interval: float = 15.0,
    ) -> AsyncIterator[ProgressEvent]:
        if replay:
            for event in self.history(operation_id):
                yield event

        queue: asyncio.Queue[ProgressEvent] = asyncio.Queue(maxsize=self._queue_size)
        async with self._lock:
            self._subscribers[operation_id].add(queue)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
                except asyncio.TimeoutError:
                    event = ProgressEvent(
                        sequence=0,
                        type=EventType.HEARTBEAT,
                        operation_id=operation_id,
                        state=OperationState.RUNNING,
                        message="heartbeat",
                    )
                yield event
        finally:
            async with self._lock:
                subscribers: Optional[Set[asyncio.Queue[ProgressEvent]]] = self._subscribers.get(operation_id)
                if subscribers is not None:
                    subscribers.discard(queue)
                    if not subscribers:
                        self._subscribers.pop(operation_id, None)

