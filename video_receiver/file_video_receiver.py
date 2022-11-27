import time
from dataclasses import dataclass
from time import sleep

import cv2

from internal_types import FrameType, IsRunningGetter

from queue_custom import Queue
from schema import FrameShape
from video_receiver.video_receiver_base import VideoReceiverConfigBase, VideoReceiverBase


@dataclass
class FileVideoReceiverConfig(VideoReceiverConfigBase):
    video_path: str


@dataclass
class FileVideoReceiver(VideoReceiverBase):
    put_queue: Queue[FrameType]

    config: FileVideoReceiverConfig

    def __post_init__(self):
        self._video_cap = cv2.VideoCapture(self.config.video_path)

        self._fps = self._video_cap.get(cv2.CAP_PROP_FPS)

    def start(self, is_running_getter: IsRunningGetter):
        is_read_success, frame = self._video_cap.read()

        sleep_time = 1 / int(self._fps)

        while is_read_success:
            t1 = time.time()
            self.put_queue.put(frame)

            is_read_success, frame = self._video_cap.read()

            delta = time.time() - t1

            sleep(sleep_time - delta)

            if not is_running_getter():
                break

    def get_fps(self) -> int:
        return int(self._fps)

    def get_shape(self) -> FrameShape:
        width = self._video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self._video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        return FrameShape(width=int(width), height=int(height))

    def shut_down(self):
        ...
