from dataclasses import dataclass
from internal_types import FrameType, IsRunningGetter

from queue_custom import Queue


@dataclass
class VideoReceiverConfigBase:
    ...


class VideoReceiverBase:
    put_queue: Queue[FrameType]

    config: VideoReceiverConfigBase

    def __init__(self):
        ...

    def start(self, is_running_getter: IsRunningGetter):
        ...

    def shut_down(self):
        ...

    def get_fps(self) -> int:
        ...

    def get_shape(self) -> int:
        ...
