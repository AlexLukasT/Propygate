import numpy as np


class Loss():
    pass


class MSE(Loss):
    def __call__(self, a, y, prime=False):
        if not prime:
            return (y - a) ** 2
        return 2 * (a - y)

    def total(self, a, y):
        return np.mean(np.sum((y - a) ** 2, axis=1), axis=0)
