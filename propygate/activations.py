import numpy as np


class ReLU:
    @staticmethod
    def __call__(x):
        return np.maximum(0, x)

    @staticmethod
    def prime(x):
        return np.heaviside(x, 0)


class Sigmoid:
    @staticmethod
    def __call__(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def prime(x):
        return Sigmoid()(x) * (1 - Sigmoid()(x))


class Linear:
    @staticmethod
    def __call__(x):
        return x

    @staticmethod
    def prime(x):
        return 1
