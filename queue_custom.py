from dataclasses import dataclass, field
from threading import Lock
from typing import Generic, TypeVar, List

M = TypeVar("M")


@dataclass
class Queue(Generic[M]):
    _list: List[M] = field(default_factory=list)

    _lock: Lock = field(default_factory=Lock)

    def get_item_count(self) -> int:
        self._lock.acquire()
        try:
            result = len(self._list)
            self._lock.release()
            return result
        except Exception as e:
            print(e)

        self._lock.release()

    def is_empty(self) -> bool:
        self._lock.acquire()
        try:
            result = len(self._list) == 0
            self._lock.release()
            return result
        except Exception as e:
            print(e)

        self._lock.release()

    def get(self) -> M:
        self._lock.acquire()
        try:
            result = self._list.pop(0)
            self._lock.release()
            return result
        except Exception as e:
            print(e)

        self._lock.release()

    def put(self, item: M) -> None:
        self._lock.acquire()
        try:
            result = self._list.append(item)
            self._lock.release()
            return result
        except Exception as e:
            print(e)

        self._lock.release()
