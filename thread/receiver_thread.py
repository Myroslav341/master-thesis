from thread.thread_base import InfiniteThread
from internal_types import IsRunningGetter
from video_receiver.video_receiver_base import VideoReceiverBase


class ReceiverThread(InfiniteThread):
    def __init__(self, receiver: VideoReceiverBase, is_running_getter: IsRunningGetter, **kwargs):
        super().__init__(is_running_getter=is_running_getter, **kwargs)
        self.receiver = receiver

    def handler(self):
        self.receiver.start(self.is_running_getter)
        print("Receiver thread is done.")
        self.receiver.shut_down()
