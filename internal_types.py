from typing import Callable, Union

from numpy import ndarray
from pygame import Surface
from pygame.surface import SurfaceType

FrameType = ndarray
IsRunningGetter = Callable[[], bool]
VideoDisplay = Union[Surface, SurfaceType]
