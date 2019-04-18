import numpy as np


def identity(x):
    return x


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


def relu(x):
    return np.maximum(x, 0)


def relu_prime(x):
    return np.heaviside(x, 0)
