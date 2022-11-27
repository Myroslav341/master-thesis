import cv2
import numpy as np
from keras_preprocessing.image import ImageDataGenerator


def augment():
    path = "data/augument"

    datagen = ImageDataGenerator(
        zoom_range=(0.9, 1.1),
        width_shift_range=0.15,
        height_shift_range=0.15,
        fill_mode='nearest'
    )

    train_x = np.ndarray(shape=(1, 128, 128, 3), dtype=float)

    for i, p in enumerate(["data/train/english_setter_132.jpg"]):
        img_data = cv2.imread(p, cv2.IMREAD_UNCHANGED)

        img_data_resized = cv2.resize(img_data, (128, 128))

        train_x[i] = np.array(img_data_resized)

    n = 0
    for batch, in datagen.flow(train_x, batch_size=5,
                              save_to_dir=path,
                              save_format='JPG'):
        n += 1
        if n > 5:
            break


if __name__ == '__main__':
    augment()
