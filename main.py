import arrow
import numpy as np
import pygame

from time import sleep
from threading import Thread
from typing import List

from matplotlib import pyplot as plt

from internal_types import FrameType
from output_video_displayer import OutputVideoDisplayer
from processor.video_processor import VideoProcessor, KValue

from queue_custom import Queue
from schema import FrameResult, InputQueueSize
from thread.displayer_thread import DisplayerThread
from thread.processor_thread import ProcessorThread
from thread.query_statistic_thread import QueryStatisticThread
from thread.receiver_thread import ReceiverThread
from video_receiver import FileVideoReceiver
from video_receiver.file_video_receiver import FileVideoReceiverConfig


_THREADS: List[Thread] = []
_IS_RUNNING: bool = False


def start_thread(thread: Thread) -> None:
    thread.start()
    sleep(0.1)

    _THREADS.append(thread)


def shut_down_threads():
    for thread in _THREADS:
        thread.join()


def is_running() -> bool:
    global _IS_RUNNING
    return _IS_RUNNING


def set_is_running(new_is_running: bool):
    global _IS_RUNNING
    _IS_RUNNING = new_is_running


def build_input_query_size_plot(statistic_thread: QueryStatisticThread):
    x_axis = np.array([((x * 10) % 1000) / 10 for x in statistic_thread.item_count_over_time.keys()])

    y_axis = np.array(list(statistic_thread.item_count_over_time.values()))

    plt.plot(x_axis, y_axis)
    plt.show()


def print_time_profiling(video_processor: VideoProcessor):
    print(video_processor._process_times)


def input_query_over_time_plot(input_query_statistic: QueryStatisticThread):
    x_axis = np.array([((x * 10) % 1000) / 10 for x in input_query_statistic.item_count_over_time.values()])
    y_axis = [arrow.get(k).timestamp() for k in input_query_statistic.item_count_over_time.keys()]
    y_axis = np.array(y_axis)

    plt.plot(y_axis, x_axis)
    plt.show()


def k_over_time_plot(video_processor: VideoProcessor):
    x_axis = np.array([((x * 10) % 1000) / 10 for x in video_processor.k_values_change_map.keys()])
    y_axis = [f"{k.up}/{k.down}" for k in video_processor.k_values_change_map.values()]
    y_axis = np.array(y_axis)

    plt.plot(x_axis, y_axis)
    plt.show()


if __name__ == '__main__':
    set_is_running(True)

    input_queue: Queue[FrameType] = Queue()
    output_queue: Queue[FrameResult] = Queue()

    input_query_size = InputQueueSize()

    video_receiver = FileVideoReceiver(
        config=FileVideoReceiverConfig(video_path="test_large.avi"),
        put_queue=input_queue,
    )
    receiver_thread = ReceiverThread(receiver=video_receiver, is_running_getter=is_running)

    SHAPE = video_receiver.get_shape()

    processor = VideoProcessor(
        input_queue=input_queue,
        output_queue=output_queue,
        input_query_size=input_query_size,
        artificial_cnn_delay_enabled=True,
        constant_k_value=False,
        k_value=KValue(1, 3),
    )
    processor_thread = ProcessorThread(
        processor=processor,
        is_running_getter=is_running,
    )

    pygame.init()
    video_display = pygame.display.set_mode((SHAPE.width, SHAPE.height))

    displayer = OutputVideoDisplayer(output_queue=output_queue, video_display=video_display, shape=SHAPE)
    displayer_thread = DisplayerThread(displayer=displayer, is_running_getter=is_running)

    input_query_statistic_thread = QueryStatisticThread(
        query=input_queue,
        is_running_getter=is_running,
        refresh_time=0.1,
        field_to_update=input_query_size,
    )

    start_thread(receiver_thread)
    start_thread(processor_thread)
    start_thread(displayer_thread)
    start_thread(input_query_statistic_thread)

    sleep(25)

    pygame.quit()

    set_is_running(False)

    shut_down_threads()
