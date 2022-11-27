from output_video_displayer import OutputVideoDisplayer
from thread.thread_base import InfiniteThread
from internal_types import IsRunningGetter


class DisplayerThread(InfiniteThread):
    def __init__(self, displayer: OutputVideoDisplayer, is_running_getter: IsRunningGetter, **kwargs):
        super().__init__(is_running_getter=is_running_getter, **kwargs)
        self.displayer = displayer

    def handler(self):
        self.displayer.display(self.is_running_getter)
        print("Display thread is done.")
