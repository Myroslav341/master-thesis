from dataclasses import dataclass, field, InitVar
from threading import Lock
from types import FrameType
from typing import Generic, TypeVar, Any, Optional


@dataclass
class BoundaryBox:
    x: int
    y: int
    width: int
    height: int

    def __str__(self):
        return f"[{self.x}; {self.y}; {self.width}; {self.height}]"

    def __repr__(self):
        return self.__str__()

    def to_list(self):
        return [self.x, self.y, self.width, self.height]


@dataclass
class FrameResult:
    frame: FrameType
    boundary_box: Optional[BoundaryBox]


@dataclass
class FrameShape:
    width: int
    height: int


T = TypeVar("T")


@dataclass
class ProtectedParam(Generic[T]):
    _lock: Lock = field(default_factory=Lock)

    field_name: str = field(default="custom_field")

    default_value: Any = field(default=None)

    def __post_init__(self):
        setattr(self, self.field_name, self.default_value)

    def get(self) -> T:
        self._lock.acquire()
        value = getattr(self, self.field_name)
        self._lock.release()
        return value

    def set(self, new_value: T) -> None:
        self._lock.acquire()
        setattr(self, self.field_name, new_value)
        self._lock.release()


@dataclass
class InputQueueSize(ProtectedParam[int]):
    default_value: int = 0
