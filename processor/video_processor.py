from dataclasses import dataclass, field
from time import sleep, time
from typing import List, Optional, Dict, Tuple

from internal_types import FrameType, IsRunningGetter
from processor.cnn_processor import CNNProcessor
from processor.interpolation_processor import InterpolationProcessor
from queue_custom import Queue
from schema import BoundaryBox, FrameResult, InputQueueSize


@dataclass
class KValue:
    up: int
    down: int

    _i: int = field(default=0)

    def is_cnn_processing(self) -> bool:
        i = self._i
        self._i = (self._i + 1) % self.down

        if i < self.up:
            return True
        return False

    def __eq__(self, other):
        return self.up == other.up and self.down == other.down


K_VALUES_LIST = [
    KValue(1, 1),
    KValue(3, 4),
    KValue(2, 3),
    KValue(1, 2),
    KValue(1, 3),
    KValue(1, 4),
    KValue(1, 5),
]

INPUT_QUERY_SIZE_CACHE_LEN = 3
INPUT_QUERY_MAX_SIZES_CACHE_LEN = 3
CNN_EXECUTION_TIMES_CACHE_LEN = 5
IDLE_TIMES_CACHE_LEN = 5
BOUNDARY_BOXES_CACHE_LEN = 3


@dataclass
class VideoProcessor:
    input_queue: Queue[FrameType]
    output_queue: Queue[FrameResult]

    input_query_size: InputQueueSize

    constant_k_value: bool = field(default=False)
    artificial_cnn_delay_enabled: bool = field(default=False)

    _cnn_processor: CNNProcessor = field(default_factory=CNNProcessor)
    _interpolation_processor: InterpolationProcessor = field(default_factory=InterpolationProcessor)

    _input_query_sizes: List[int] = field(default_factory=lambda: [0 for _ in range(INPUT_QUERY_SIZE_CACHE_LEN)])
    _input_query_max_sizes: List[int] = field(default_factory=lambda: [0 for _ in range(INPUT_QUERY_MAX_SIZES_CACHE_LEN)])

    _frame_processed_count: int = field(default=0)

    _update_query_max_size_after__frame_count: int = field(default=5)
    _freeze_k_value_after_update__frame_count: int = field(default=15)

    k_value: KValue = KValue(1, 2)

    _k_values_list: List[KValue] = field(default_factory=lambda: K_VALUES_LIST)
    _k_value_updated_on__frame_index: int = field(default=0)
    k_values_change_map: Dict[float, KValue] = field(default_factory=dict)

    _cnn_execution_times: List[float] = field(default_factory=lambda: [0 for _ in range(CNN_EXECUTION_TIMES_CACHE_LEN)])
    _idle_times: List[float] = field(default_factory=lambda: [0 for _ in range(IDLE_TIMES_CACHE_LEN)])

    _boundary_boxes: List[BoundaryBox] = field(default_factory=list)

    _idle_start_time: Optional[float] = field(default=None)
    _idle_sleep_time: float = field(default=0.01)

    _process_times: List[Tuple[float, float]] = field(default_factory=list)

    def start_processing(self, is_running_getter: IsRunningGetter):
        execution_start_time = time()

        while True:
            if not is_running_getter():
                break

            if self.input_queue.is_empty():
                self._system_is_idle_status()
                continue

            # processing start
            process_time = time()

            # system was idle
            if self._idle_start_time:
                self._idle_times = [(time() - self._idle_start_time), *self._idle_times[:-1]]
                self._idle_start_time = None

            self._update_query_size()

            if not self.constant_k_value:
                self._check_k_value()

            self.k_values_change_map[time() - execution_start_time] = self.k_value

            frame = self.input_queue.get()

            boundary_box = self._process_frame(frame)

            self._frame_processed_count += 1

            self._process_times.append((process_time, time()))

            self.output_queue.put(
                FrameResult(
                    frame=frame,
                    boundary_box=boundary_box,
                )
            )

            # processing end(-1)

    def _process_frame(self, frame: FrameType) -> BoundaryBox:
        if self.k_value.is_cnn_processing():
            return self._process_frame_with_cnn(frame=frame)

        return self._process_frame_with_interpolation()

    def _process_frame_with_cnn(self, frame: FrameType) -> BoundaryBox:
        cnn_start_time = time()

        if self.artificial_cnn_delay_enabled and 200 < self._frame_processed_count < 400:
            sleep(0.02)

        boundary_box = self._cnn_processor.predict(frame)

        self._cnn_execution_times = [(time() - cnn_start_time), *self._cnn_execution_times[:-1]]

        self._update_boundary_boxes(boundary_box)

        return boundary_box

    def _process_frame_with_interpolation(self) -> Optional[BoundaryBox]:
        boundary_box = self._interpolation_processor.predict(boundary_boxes=self._boundary_boxes)

        if not boundary_box:
            return boundary_box

        self._update_boundary_boxes(boundary_box)

        return boundary_box

    def _update_boundary_boxes(self, new_boundary_box: BoundaryBox) -> None:
        if len(self._boundary_boxes) < BOUNDARY_BOXES_CACHE_LEN:
            self._boundary_boxes.append(new_boundary_box)
        else:
            self._boundary_boxes = [new_boundary_box, *self._boundary_boxes[:-1]]

    def _system_is_idle_status(self):
        if not self._idle_start_time:
            self._idle_start_time = time()

        self._update_query_size(0)

        sleep(self._idle_sleep_time)

    def _check_k_value(self) -> None:
        is_k_value_update_on_freeze = (
            self._k_value_updated_on__frame_index + self._freeze_k_value_after_update__frame_count > self._frame_processed_count
        )
        if is_k_value_update_on_freeze:
            return None

        is_input_query_not_empty = self._input_query_sizes[0] > 2 and self._input_query_sizes[1] > 2
        is_input_query_empty = self._input_query_sizes[0] < 2 and self._input_query_sizes[1] < 2

        # todo: block going up unless IDLE time or CNN times changes (freeze them here).
        if is_input_query_not_empty:
            self._k_value_updated_on__frame_index = self._frame_processed_count

            # we are going down already, so no need to decrease K value
            if self._is_input_query_goes_down():
                return None

            self._decrease_k_value()

            return None

        cnn_execution_time_avg = sum(self._cnn_execution_times) / len(self._cnn_execution_times)
        idle_time_avg = sum(self._idle_times) / len(self._idle_times)

        # todo: add condition on IDLE and CNN time changed
        if is_input_query_empty and cnn_execution_time_avg / 2 < idle_time_avg:
            self._k_value_updated_on__frame_index = self._frame_processed_count
            self._increase_k_value()

    def _increase_k_value(self):
        index = self._k_values_list.index(self.k_value)

        if index == 0:
            return self.k_value

        self.k_value = self._k_values_list[index - 1]

    def _decrease_k_value(self):
        index = self._k_values_list.index(self.k_value)

        if index + 1 == len(self._k_values_list):
            return self.k_value

        self.k_value = self._k_values_list[index + 1]

    def _is_input_query_goes_down(self) -> bool:
        is_query_goes_down = self._input_query_max_sizes[0] < self._input_query_max_sizes[1] or self._input_query_max_sizes[1] < self._input_query_max_sizes[2]

        return is_query_goes_down

    def _update_query_size(self, current_size: Optional[int] = None):
        if not current_size:
            current_size = self.input_query_size.get()

        self._input_query_sizes = [current_size, *self._input_query_sizes[:-1]]

        if self._frame_processed_count % self._update_query_max_size_after__frame_count != 0:
            return None

        self._input_query_max_sizes = [max(self._input_query_sizes), *self._input_query_max_sizes[:-1]]
