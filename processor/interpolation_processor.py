from dataclasses import dataclass
from typing import List, Optional

from schema import BoundaryBox


@dataclass
class InterpolationProcessor:
    def predict(self, boundary_boxes: List[BoundaryBox]) -> Optional[BoundaryBox]:
        if len(boundary_boxes) < 2:
            return None

        delta_x = self._calc_delta(boundary_boxes, lambda x: x.x)
        delta_y = self._calc_delta(boundary_boxes, lambda x: x.y)
        delta_width = self._calc_delta(boundary_boxes, lambda x: x.width)
        delta_height = self._calc_delta(boundary_boxes, lambda x: x.height)

        return BoundaryBox(
            x=boundary_boxes[0].x + delta_x / 3,
            y=boundary_boxes[0].y + delta_y / 3,
            width=boundary_boxes[0].width + delta_width / 3,
            height=boundary_boxes[0].height + delta_height / 3,
        )

    def _calc_delta(self, boundary_boxes: List[BoundaryBox], getter):
        delta = []
        for i in range(len(boundary_boxes) - 1):
            p1 = getter(boundary_boxes[i])
            p2 = getter(boundary_boxes[i + 1])
            delta.append(p1 - p2)

        return sum(delta) / len(delta)
