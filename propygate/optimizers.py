import numpy as np


class Optimizer:

    def __init__(self, lr):

        self.lr = lr


class GradientDescent(Optimizer):

    def __init__(self, lr):

        super().__init__(lr)

    def update(self, weights, gradients_w, gradients_b):

        new_weights = weights

        for i in range(len(new_weights)):
            new_weights[i][0] -= self.lr * gradients_w[i]
            new_weights[i][1] -= self.lr * gradients_b[i]

        return new_weights
