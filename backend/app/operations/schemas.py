from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:
    class _FieldInfo:
        def __init__(self, default: Any = None, default_factory: Optional[Any] = None) -> None:
            self.default = default
            self.default_factory = default_factory

        def value(self) -> Any:
            if self.default_factory is not None:
                return self.default_factory()
            return self.default

    def Field(default: Any = None, **kwargs: Any) -> _FieldInfo:
        return _FieldInfo(default=default, default_factory=kwargs.get("default_factory"))

    class BaseModel:
        def __init__(self, **data: Any) -> None:
            fields: Dict[str, Any] = {}
            for cls in reversed(type(self).__mro__):
                fields.update(getattr(cls, "__annotations__", {}))

            for name in fields:
                if name in data:
                    value = data.pop(name)
                else:
                    default = getattr(type(self), name, None)
                    value = default.value() if isinstance(default, _FieldInfo) else default
                setattr(self, name, value)

            for name, value in data.items():
                setattr(self, name, value)

        def dict(self) -> Dict[str, Any]:
            return dict(self.__dict__)

        def model_dump(self, mode: Optional[str] = None) -> Dict[str, Any]:
            return dict(self.__dict__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OperationKind(str, Enum):
    SCAN = "scan"
    INDEX = "index"
    SEARCH = "search"
    ANALYZE = "analyze"
    CUSTOM = "custom"


class OperationState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class EventType(str, Enum):
    SNAPSHOT = "snapshot"
    PROGRESS = "progress"
    HEARTBEAT = "heartbeat"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class AppConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:1420",
            "http://127.0.0.1:1420",
        ]
    )


class OperationRequest(BaseModel):
    kind: OperationKind = OperationKind.CUSTOM
    params: Dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    source: Optional[str] = None
    replay_dir: Optional[str] = None
    force: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClipSearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    filters: Dict[str, Any] = Field(default_factory=dict)


class OperationStatus(BaseModel):
    operation_id: str
    kind: str
    state: OperationState
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=utc_now)
    finished_at: Optional[datetime] = None


class OperationStartResponse(BaseModel):
    operation_id: str
    status_url: str
    events_url: str
    state: OperationState


class ProgressEvent(BaseModel):
    sequence: int
    type: EventType
    operation_id: str
    state: OperationState
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = ""
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=utc_now)
