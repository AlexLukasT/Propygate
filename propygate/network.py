from propygate import utils, layers
import numpy as np
import matplotlib.pyplot as plt
import os


class NeuralNetwork:
    def __init__(self):
        # if len(neurons) < 2:
        #     raise ValueError("Need atleast one input and one output layer")

        # self.neurons = neurons
        # self.num_weights = len(self.neurons) - 1

        # self.weights = []
        # self.bias = []

        self.layers = []

        self._initialized = False

    def add(self, layer):
        if not isinstance(layer, layers.Layer):
            raise TypeError("Pass in a valid layer, not a {}".format(type(layer)))
        if not self.layers and layer.input_shape is None:
            raise ValueError("The first layer of a network needs an input shape")
        self.layers.append(layer)

    def initialize(self, loss):
        for i in range(len(self.layers)):
            previous_shape = None if i == 0 else self.layers[i - 1].num_neurons
            self.layers[i]._build(previous_shape)

        self.loss = loss
        self._initialized = True

    # def forwardprop(self, inp):
    # activations = [inp]
    # zs = []
    # for i in range(self.num_weights):
    #     z = np.dot(self.weights[i], activations[-1]) + self.bias[i]
    #     zs.append(z)
    #     activations.append(self.sigmoid(z))
    # return zs, activations

    # def backprop(self, activations, y):
    # gradients_w = [np.empty(w.shape) for w in self.weights]
    # gradients_b = [np.empty(b.shape) for b in self.bias]

    # h = 2 * (activations[-1] - y)  # MSE
    # for i in reversed(range(self.num_weights)):
    #     h *= self.sigmoid_prime(zs[i])
    #     gradients_b[i] = h
    #     gradients_w[i] = np.outer(h, activations[i].T)
    #     h = np.dot(self.weights[i].T, h)

    # return gradients_w, gradients_b

    def _forwardprop(self, inp):
        activations = [inp]
        for i in range(len(self.layers)):
            activation = self.layers[i]._forwardprop(activations[i])
            activations.append(activation)
        return activations

    def _backprop(self, activations, y):
        gradients_w = [np.empty(l.weights.shape) for l in self.layers]
        gradients_b = [np.empty(l.bias.shape) for l in self.layers]

        h = self.loss.prime(activations[-1], y)
        for i in reversed(range(len(self.layers))):
            new_h, gradient_w, gradient_b = self.layers[i]._backprop(h, activations[i])
            gradients_w[i] = gradient_w
            gradients_b[i] = gradient_b
            h = new_h
        return gradients_w, gradients_b

    def _get_gradients(self, x_batch, y_batch):
        gradients_w = [np.zeros(l.weights.shape) for l in self.layers]
        gradients_b = [np.zeros(l.bias.shape) for l in self.layers]

        for i in range(len(x_batch)):  # loop over one batch
            activations = self._forwardprop(x_batch[i])
            d_grads_w, d_grads_b = self._backprop(activations, y_batch[i])
            gradients_w = [gw + dgw for gw, dgw in zip(gradients_w, d_grads_w)]
            gradients_b = [gb + dgb for gb, dgb in zip(gradients_b, d_grads_b)]

        return gradients_w, gradients_b

    def output(self, x):
        return self._forwardprop(x)[-1]

    @staticmethod
    def predict(self, x):
        return self.model_output(x) / np.sum(self.output(x))
