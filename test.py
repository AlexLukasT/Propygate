from propygate import network, datasets, utils, layers, optimizers, losses
import numpy as np


def main():

    model = network.NeuralNetwork()
    model.add(layers.FullyConnected(128, activation="ReLU", input_shape=784))
    model.add(layers.FullyConnected(64, activation="ReLU"))
    model.add(layers.FullyConnected(32, activation="ReLU"))
    model.add(layers.FullyConnected(10, activation="Sigmoid"))

    loss = losses.MeanSquaredError()
    model.initialize(loss)

    optimizer = optimizers.GradientDescent(model, learning_rate=0.005, batch_size=50)

    (x_train, x_test, x_valid), (y_train, y_test, y_valid) = datasets.load_mnist()

    x_train = np.reshape(x_train, (x_train.shape[0], 28 * 28))  # flatten images
    x_test = np.reshape(x_test, (x_test.shape[0], 28 * 28))
    x_valid = np.reshape(x_valid, (x_valid.shape[0], 28 * 28))

    optimizer.train(
        x_train[:100], y_train[:100], epochs=10, valid_data=(x_valid, y_valid)
    )

    # for i in range(10):
    #     utils.plot_example(
    #         x_test[i].reshape(28, 28),
    #         model.predict(x_test[i]),
    #         "examples/example_{}.png".format(i + 1),
    #     )


if __name__ == "__main__":
    main()
