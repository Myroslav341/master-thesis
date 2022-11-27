from dataclasses import dataclass
from time import sleep

import numpy as np
import pygame

from internal_types import IsRunningGetter, VideoDisplay
from queue_custom import Queue
from schema import FrameResult, FrameShape


@dataclass
class OutputVideoDisplayer:
    video_display: VideoDisplay
    shape: FrameShape

    output_queue: Queue[FrameResult]

    def display(self, is_running_getter: IsRunningGetter):
        while True:
            if not is_running_getter():
                break

            if self.output_queue.is_empty():
                sleep(0.01)
                continue

            frame = self.output_queue.get()

            boundary_box = frame.boundary_box
            frame = frame.frame

            surf = pygame.surfarray.make_surface(np.flipud(np.rot90(frame)))

            self.video_display.blit(surf, (0, 0))
            color = (255, 0, 0)

            if boundary_box:
                pygame.draw.rect(self.video_display, color, pygame.Rect(
                    boundary_box.x * self.shape.width,
                    boundary_box.y * self.shape.height,
                    boundary_box.width * self.shape.width,
                    boundary_box.height * self.shape.height,
                ), width=3)

            pygame.display.update()
