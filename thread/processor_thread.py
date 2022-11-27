from processor.video_processor import VideoProcessor
from thread.thread_base import InfiniteThread
from internal_types import IsRunningGetter


class ProcessorThread(InfiniteThread):
    def __init__(self, processor: VideoProcessor, is_running_getter: IsRunningGetter, **kwargs):
        super().__init__(is_running_getter=is_running_getter, **kwargs)
        self.processor = processor

    def handler(self):
        self.processor.start_processing(self.is_running_getter)
        print("Processor thread is done.")
