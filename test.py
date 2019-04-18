from propygate.model import FeedForward
from propygate.layers import FullyConnected
import numpy as np


def main():

    model = FeedForward()
    model.add(FullyConnected(20, activation="relu", input_dim=5))
    model.add(FullyConnected(30, activation="relu"))
    model.add(FullyConnected(10, activation="sigmoid"))

    model.initialize()

    model.layers[0]._fprop(np.ones(5))


if __name__ == "__main__":
    main()
