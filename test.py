from propygate import network, optimizers, losses, datasets, utils
import numpy as np


def main():

    model = network.NeuralNetwork([784, 32, 10])

    model.build()

    (x_train, x_test, x_valid), (y_train, y_test, y_valid) = datasets.load_mnist()

    x_train = np.reshape(x_train, (x_train.shape[0], 28*28))  # flatten images

    model.train(x_train[:10000], y_train[:10000], epochs=10, batch_size=64)


if __name__ == "__main__":
    main()
