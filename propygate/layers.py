import numpy as np
from propygate import activations


class Layer:
    pass


class FullyConnected(Layer):
    def __init__(self, num_neurons, activation=None, input_shape=None):
        self.num_neurons = num_neurons
        self.activation = self.handle_activation(activation)
        self.input_shape = input_shape

        self.weights = None
        self.bias = None
        self._z_cache = None

    def handle_activation(self, activation):
        if activation is None:
            return activations.Linear()
        if isinstance(activation, str):
            try:
                activation = getattr(activations, activation)()
            except AttributeError:
                raise ValueError("Unknown activation function {}".format(activation))
            return activation
        return activation

    def _build(self, previous_shape=None):
        if self.input_shape is not None:
            self.weights = np.random.uniform(
                -1, 1, (self.num_neurons, self.input_shape)
            )
        else:
            self.weights = np.random.uniform(-1, 1, (self.num_neurons, previous_shape))
        self.bias = np.random.uniform(0, 1, self.num_neurons)

    def _forwardprop(self, inp):
        z = np.dot(self.weights, inp) + self.bias
        self._z_cache = z
        return self.activation(z)

    def _backprop(self, h, activation):
        h *= self.activation.prime(self._z_cache)
        gradient_b = h
        gradient_w = np.outer(h, activation.T)
        h = np.dot(self.weights.T, h)
        return h, gradient_w, gradient_b

    def __repr__(self):
        return "<FullyConnected neurons={}>".format(self.num_neurons)
