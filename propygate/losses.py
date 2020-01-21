import numpy as np


class MeanSquaredError:
    @staticmethod
    def prime(output, y):
        return 2 * (output - y)

    @staticmethod
    def total(outputs, y):
        return np.mean(np.sum((outputs - y) ** 2, axis=1), axis=0)
