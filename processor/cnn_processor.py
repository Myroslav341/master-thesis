import cv2
import numpy as np

from dataclasses import dataclass

from keras.saving.save import load_model

from internal_types import FrameType
from schema import BoundaryBox


@dataclass
class CNNProcessor:
    model_name: str = "models/main.h5"

    def __post_init__(self):
        self.model = load_model(self.model_name)

    def predict(self, frame: FrameType) -> BoundaryBox:
        size = 128
        batch_x = np.zeros(shape=(1, size, size, 3), dtype=float)

        img_data_resized = cv2.resize(frame, (size, size))
        batch_x[0] = img_data_resized

        batch_x = batch_x.astype("float32") / 255

        result = self.model.predict(batch_x)

        result = result[0]

        return BoundaryBox(
            x=result[0],
            y=result[1],
            width=result[2],
            height=result[3],
        )
