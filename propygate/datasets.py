from tensorflow import keras
import numpy as np


def load_mnist(test_split=0.15, validation_split=0.05):

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    x_norm = x / 255  # normalizw to 0 - 1

    n = x_norm.shape[0]
    y_categorical = np.zeros((n, 10))
    y_categorical[np.arange(n), y] = 1

    num_train = int(n * (1 - test_split - validation_split))
    num_train_plus_test = int(n * (1 - validation_split))
    indices = [num_train+1, num_train_plus_test]

    x_train, x_test, x_valid = np.split(x_norm, indices)
    y_train, y_test, y_valid = np.split(y_categorical, indices)

    return (x_train, x_test, x_valid), (y_train, y_test, y_valid)
