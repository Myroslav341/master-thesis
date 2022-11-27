import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # plt.plot(
    #     np.array(
    #         [
    #             0.3,
    #             0.43,
    #             0.57,
    #             0.62,
    #             0.71,
    #             0.74,
    #             0.72,
    #             0.78,
    #             0.80,
    #             0.82,
    #
    #             0.85,
    #             0.86,
    #             0.83,
    #             0.85,
    #             0.84,
    #             0.87,
    #             0.83,
    #             0.84,
    #             0.86,
    #             0.85,
    #
    #             0.84,
    #             0.87,
    #             0.83,
    #             0.84,
    #             0.86,
    #             0.82,
    #             0.84,
    #             0.85,
    #             0.86,
    #             0.87,
    #         ]
    #     )
    # )
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train'], loc='upper left')
    # plt.show()

    # Plot training & validation loss values
    plt.plot(
        np.array(
            [
                0.605,
                0.574,
                0.526,
                0.506,
                0.490,
                0.461,
                0.449,
                0.449,
                0.435,
                0.430,

                0.432,
                0.429,
                0.421,
                0.418,
                0.417,
                0.418,
                0.413,
                0.409,
                0.403,
                0.408,

                0.410,
                0.412,
                0.411,
                0.410,
                0.412,
                0.411,
                0.403,
                0.399,
                0.396,
                0.388,
            ]
        )
    )
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()
