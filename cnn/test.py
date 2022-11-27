import tensorflow as tf
from keras import Input, Model
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils.layer_utils import print_summary

if __name__ == '__main__':
    vgg = tf.keras.applications.VGG16(weights="imagenet", include_top=False,
                                      input_tensor=Input(shape=[128, 128, 3]))
    # freeze all VGG layers so they will *not* be updated during the
    vgg.trainable = True
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

    print_summary(model)
    # plot_model(model, "1.png", show_shapes=True, expand_nested=False, show_layer_activations=True)
