from propygate import network, layers, optimizers, losses, datasets, utils
import numpy as np


def main():

    model = network.FeedForward()
    model.add(layers.FullyConnected(20, activation="relu", input_dim=(28, 28)))
    model.add(layers.FullyConnected(30, activation="relu"))
    model.add(layers.FullyConnected(10, activation="sigmoid"))

    optimizer = optimizers.GradientDescent(1e-3)
    loss = losses.MSE()
    model.initialize(optimizer, loss)

    print(model)

    (x_train, x_test, x_valid), (y_train, y_test, y_valid) = datasets.load_mnist()

    model.train(x_train[:1000], y_train[:1000], epochs=10, batch_size=64,
                val_data=(x_valid, y_valid))
    model.plot_metrics()

    prediction = model.predict(x_test[0])
    utils.plot_example(x_test[0], prediction)


if __name__ == "__main__":
    main()
