import cv2
import keras
import tensorflow as tf
from keras import Model, Sequential, Input
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.saving.save import load_model
from keras.utils import plot_model
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy import array
from tensorflow import TensorSpec

BATCH_SIZE = 32
EPOCH_SIZE = 20


def get_model():
    vgg = tf.keras.applications.VGG16(weights="imagenet", include_top=False,
                input_tensor=Input(shape=[128, 128, 3]))
    # freeze all VGG layers so they will *not* be updated during the
    vgg.trainable = False
    # training process
    # flatten the max-pooling output of VGG

    flatten = vgg.output
    flatten = Flatten()(flatten)
    # construct a fully-connected layer header to output the predicted
    # bounding box coordinates
    bboxHead = Dense(2048, activation="relu")(flatten)
    bboxHead = Dense(1024, activation="relu")(bboxHead)
    bboxHead = Dropout(0.2)(bboxHead)
    bboxHead = Dense(128, activation="relu")(bboxHead)
    bboxHead = Dense(4, activation="sigmoid")(bboxHead)
    # construct the model we will fine-tune for bounding box regression
    model = Model(inputs=vgg.input, outputs=bboxHead)

    opt = Adam(lr=1e-4)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['acc'])

    return model

    # reduce Adam(lr=1e-5), other activation on layers, try pre-trained
    # mse -0.5

    # return model2
    # vgg = tf.keras.applications.VGG16(input_shape=[128, 128, 3], include_top=False, weights='imagenet')
    # x = Flatten()(vgg.output)
    # x = Dense(1024, activation='sigmoid')(x)
    # model1 = Model(vgg.input, x)
    #
    # x = Dropout(0.2)(x)
    # model1 = Model(model1.input, x)
    #
    # x_2 = Dense(4, activation='sigmoid')(x)
    # model2 = Model(model1.input, x_2)
    #
    # model2.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])
    # # plot the model
    # plot_model(model2, "first_model.png", show_shapes=True, expand_nested=False)
    #
    # return model2


def synthetic_gen(batch_size: int = BATCH_SIZE):
    train_data_file = open("cnn/train_data.txt")

    lines = train_data_file.readlines()
    line_index = 0

    train_data_file.close()

    size = 128

    while True:
        batch_x = np.zeros(shape=(batch_size, size, size, 3), dtype=float)
        batch_y = np.zeros((batch_size, 4))

        for index in range(batch_size):
            line_index += 1
            line_index = line_index % len(lines)

            if line_index == 0:
                print("start new sequence")

            image = lines[line_index]

            image_path = image.split(",")[0].strip()
            img_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            img_data_resized = cv2.resize(img_data, (size, size))
            batch_x[index] = img_data_resized

            box = image.split(",")[2].strip().strip("[]")
            sizes = box.split(";")

            batch_y[index, 0] = int(sizes[0].strip()) / img_data.shape[1]
            batch_y[index, 1] = int(sizes[1].strip()) / img_data.shape[0]
            batch_y[index, 2] = (int(sizes[2].strip()) - int(sizes[0].strip())) / img_data.shape[1]
            batch_y[index, 3] = (int(sizes[3].strip()) - int(sizes[1].strip())) / img_data.shape[0]

        batch_x = batch_x.astype("float32") / 255

        yield batch_x, batch_y


def get_validate_data():
    train_data_file = open("cnn/validate_data.txt")

    lines = train_data_file.readlines()
    line_index = 0

    batch_size = len(lines)

    train_data_file.close()

    size = 128

    batch_x = np.zeros(shape=(batch_size, size, size, 3), dtype=float)
    batch_y = np.zeros((batch_size, 4))

    for index in range(batch_size):
        line_index += 1
        line_index = line_index % len(lines)

        if line_index == 0:
            print("start new sequence")

        image = lines[line_index]

        image_path = image.split(",")[0].strip()
        img_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img_data_resized = cv2.resize(img_data, (size, size))
        batch_x[index] = img_data_resized

        box = image.split(",")[2].strip().strip("[]")
        sizes = box.split(";")

        batch_y[index, 0] = int(sizes[0].strip()) / img_data.shape[1]
        batch_y[index, 1] = int(sizes[1].strip()) / img_data.shape[0]
        batch_y[index, 2] = (int(sizes[2].strip()) - int(sizes[0].strip())) / img_data.shape[1]
        batch_y[index, 3] = (int(sizes[3].strip()) - int(sizes[1].strip())) / img_data.shape[0]

    batch_x = batch_x.astype("float32") / 255

    return batch_x, batch_y


def train(model):
    h = model.fit_generator(
        synthetic_gen(),
        steps_per_epoch=EPOCH_SIZE,
        epochs=20,
        validation_data=get_validate_data(),
    )
    return h


def plot_predict(model_trained):
    # generate new image
    x, _ = next(synthetic_gen(batch_size=1))
    # predict
    result = model_trained.predict(x)

    print(result)

    result = result[0]

    result = [
        (result[0] + 0.5) * x.shape[2],
        (result[1] + 0.5) * x.shape[1],
        (result[2] + 0.5) * x.shape[2],
        (result[3] + 0.5) * x.shape[1],
    ]

    fig, ax = plt.subplots(1)
    ax.imshow(x[0] + 0.5)
    rect = Rectangle(xy=(result[0], result[1]), width=result[2], height=result[3], linewidth=1,edgecolor='g',facecolor='none')
    ax.add_patch(rect)
    plt.show()


def plot_predict_new(model_trained, data):
    # predict
    result = model_trained.predict(data)

    print(result)
    print(data.shape)

    result = result[0]

    result = [
        result[0] * data.shape[2],
        result[1] * data.shape[1],
        result[2] * data.shape[2],
        result[3] * data.shape[1],
    ]

    fig, ax = plt.subplots(1)
    ax.imshow(data[0])
    rect = Rectangle(xy=(result[0], result[1]), width=result[2], height=result[3], linewidth=3,edgecolor='g',facecolor='none')
    ax.add_patch(rect)
    plt.show()


def visual():
    a = synthetic_gen(1)
    for _ in range(10):
        x, y = next(a)

        result = y[0]

        result = [
            result[0] * x.shape[2],
            result[1] * x.shape[1],
            result[2] * x.shape[2],
            result[3] * x.shape[1],
        ]

        fig, ax = plt.subplots(1)
        ax.imshow(x[0])
        rect = Rectangle(xy=(result[0], result[1]), width=result[2], height=result[3],
                         linewidth=3, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        plt.show()


def get_test_data():
    batch_size = 1
    train_data_file = open("cnn/test.txt")

    lines = train_data_file.readlines()
    line_index = 0

    print(len(lines))

    train_data_file.close()

    size = 128

    while True:
        batch_x = np.zeros(shape=(batch_size, size, size, 3), dtype=float)

        for index in range(batch_size):
            line_index += 1
            print(line_index)
            if line_index >= len(lines):
                raise StopIteration()

            image = lines[line_index]

            image_path = image.split(",")[0].strip()
            img_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            img_data_resized = cv2.resize(img_data, (size, size))
            batch_x[index] = img_data_resized

        batch_x = batch_x.astype("float32") / 255

        yield batch_x


if __name__ == '__main__':
    # getter = get_test_data()
    # model = load_model('model_086.h5')
    #
    # while True:
    #     try:
    #         data = next(getter)
    #     except StopIteration:
    #         break
    #     # visual()
    #
    #     result = model.predict(data)
    #
    #     print(result)
    #     print(data.shape)
    #
    #     result = result[0]
    #
    #     result = [
    #         result[0] * data.shape[2],
    #         result[1] * data.shape[1],
    #         result[2] * data.shape[2],
    #         result[3] * data.shape[1],
    #     ]
    #
    #     fig, ax = plt.subplots(1)
    #     ax.imshow(data[0])
    #     rect = Rectangle(xy=(result[0], result[1]), width=result[2], height=result[3], linewidth=3,edgecolor='g',facecolor='none')
    #     ax.add_patch(rect)
    #     plt.show()

    model = get_model()
    h = train(model)
    # model.save('model_new.h5')
    # a = synthetic_gen()
    # while True:
    #     x, _ = next(a)
    #     plot_predict_new(model, x)

    plt.plot(h.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(h.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()
