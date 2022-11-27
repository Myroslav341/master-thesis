import os
import cv2
import keras
from keras.utils.layer_utils import print_summary
from keras_preprocessing.image import ImageDataGenerator
from numpy import array, ndarray
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from typing import List
import matplotlib.pyplot as plt


class CNN:
    def train(self, save_name: str = None):

        x_train, y_train = self._load_train_data()

        print(x_train[0])
        print(y_train[0])

        return

        x_train, y_train = self._load_train_data()
        x_validate, y_validate = self.__load_data(validate_path)
        x_test, y_test = self.__load_data(test_path)

        batch_size = 128
        self.num_classes = 2
        epochs = 30

        img_rows, img_cols = 64, 64

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_validate = x_validate.reshape(x_validate.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            self.input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_validate = x_validate.reshape(x_validate.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            self.input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_validate = x_validate.astype('float32')
        x_test = x_test.astype('float32')

        x_train /= 255
        x_validate /= 255
        x_test /= 255

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_validate.shape[0], 'test samples')

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_validate = keras.utils.to_categorical(y_validate, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        model = self.__create_model()

        print_summary(model)

        aug = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest"
        )

        h = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                                epochs=epochs,
                                steps_per_epoch=len(x_train) // batch_size,
                                validation_data=(x_validate, y_validate))

        score = model.evaluate(x_test, y_test, verbose=0)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # if save_name is not None:
        #     model.save(Config.BASE_DIR + '/cnn/models/' + save_name + '.h5')

        print_summary(model)

        plt.plot(h.history['acc'])
        plt.plot(h.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(h.history['loss'])
        plt.plot(h.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def load(self, model_name: str):
        self.model = load_model(Config.BASE_DIR + '/cnn/models/' + model_name)

    def predict(self, file_path: str) -> List[float]:
        img_data = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (64, 64))

        img_data = img_data.reshape(1, 64, 64, 1)

        img_data = img_data.astype('float32')

        img_data /= 255

        f = self.model.predict(img_data)

        return f

    def __create_model(self):
        model = Sequential()
        model.add(Conv2D(8, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))

        model.add(Conv2D(16, (3, 3),
                         activation='relu'
                         ))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3),
                         activation='relu'
                         ))

        model.add(Conv2D(64, (3, 3),
                         activation='relu'
                         ))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3),
                         activation='relu'
                         ))

        model.add(Conv2D(128, (3, 3),
                         activation='relu'
                         ))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(1024,
                        activation='relu'
                        ))
        model.add(Dropout(0.5))
        model.add(Dense(512,
                        activation='relu'
                        ))
        model.add(Dense(256,
                        activation='relu'
                        ))
        model.add(Dense(64,
                        activation='relu'
                        ))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['acc'])

        return model

    def _load_train_data(self):
        f = open("train_data.txt")

        lines = f.readlines()

        f.close()

        train_pic = len(lines)

        data = ndarray(shape=(train_pic, 64, 64), dtype=float)
        labels = array(lines)

        for i, image in enumerate(lines):
            image_path = image.split(",")[0].strip()
            img_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (64, 64))
            data[i] = array(img_data)

            box = image.split(",")[2].strip().strip("[]")
            sizes = box.split(";")

            labels[i] = [
                int(sizes[0].strip()),
                int(sizes[1].strip()),
                int(sizes[2].strip()),
                int(sizes[3].strip()),
            ]

        return data, labels

    def _load_data(self, path):
        train_pic = os.listdir(path)

        data = ndarray(shape=(len(train_pic), 64, 64), dtype=float)
        labels = array(train_pic)

        for i, x in enumerate(train_pic):
            img_data = cv2.imread(path + '\\' + x, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.resize(img_data, (64, 64))
            data[i] = array(img_data)

            if 'rect' in x:
                labels[i] = 0
            else:
                labels[i] = 1

        return data, labels
