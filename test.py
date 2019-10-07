from propygate import network, datasets
import numpy as np


def main():

    model = network.NeuralNetwork([784, 64, 64, 10])

    model.build()

    (x_train, x_test, x_valid), (y_train, y_test, y_valid) = datasets.load_mnist()

    x_train = np.reshape(x_train, (x_train.shape[0], 28*28))  # flatten images

    model.train(x_train, y_train, epochs=10, batch_size=64)


if __name__ == "__main__":
    main()
