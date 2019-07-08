import numpy as np


class Optimizer:

    def __init__(self, lr):

        self.lr = lr


class StochasticGradientDescent(Optimizer):

    def __init__(self, lr):

        super().__init__(lr)
