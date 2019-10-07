from propygate import network, datasets, utils
import numpy as np


def main():

    model = network.NeuralNetwork([784, 64, 64, 10])

    model.build()

    (x_train, x_test, x_valid), (y_train, y_test, y_valid) = datasets.load_mnist()

    x_train = np.reshape(x_train, (x_train.shape[0], 28*28))  # flatten images
    x_test = np.reshape(x_test, (x_test.shape[0], 28*28))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], 28*28))

    model.train(x_train, y_train, epochs=30, batch_size=64,
                valid_data=(x_valid, y_valid))

    for i in range(10):
        utils.plot_example(x_test[i].reshape(28, 28),
                           model.predict(x_test[i]),
                           "examples/example_{}.png".format(i+1))


if __name__ == "__main__":
    main()
