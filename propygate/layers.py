import numpy as np
import propygate.activations as activations


class FullyConnected:

    def __init__(self, num_neurons, activation=None, input_dim=None):

        self.num_neurons = num_neurons
        self._activation = activation if activation else "identity"
        self.input_dim = input_dim

        self.type = None
        self._weights = None
        self._bias = None

        self._fprop_cache = None
        self._gradient_w = None
        self._gradient_b = None

    def activation(self, x, prime=False):

        if not prime:
            activation_func = getattr(activations, self._activation)
        else:
            activation_func = getattr(activations, self._activation + "_prime")

        return activation_func(x)

    def _initialize_weights(self, previous_neurons=None):

        if self.type == "input" and previous_neurons is None:
            if isinstance(self.input_dim, int):
                size = (self.input_dim, self.num_neurons)
            elif isinstance(self.input_dim, (list, tuple, np.ndarray)):
                size = (np.prod(self.input_dim), self.num_neurons)
            else:
                raise TypeError("Invalid input dim: pass in a list, tuple or numpy array")

        else:
            size = (previous_neurons, self.num_neurons)

        self._weights = np.random.uniform(-1, 1, size=size)
        self._bias = np.zeros(self.num_neurons)

    def _fprop(self, x):

        if isinstance(self.input_dim, (list, tuple, np.ndarray)):
            x = x.flatten()
        z = np.dot(x, self._weights) + self._bias
        self._fprop_cache = z

        return self.activation(z)

    def _bprop(self, h, a_l, prev_weights):
        z = self._fprop_cache
        h *= self.activation(z, prime=True)
        self.gradient_b = h
        self.gradient_w = np.outer(a_l, h)
        return np.dot(prev_weights, h)

    def get_weights(self):
        return self._weights, self._bias

    def set_weights(self, new_weights, new_bias):
        if not isinstance(new_weights, np.ndarray):
            raise TypeError("weights must be a numpy array")

        if not isinstance(new_bias, np.ndarray):
            raise TypeError("weights must be a numpy array")

        if not new_weights.shape == self._weights.shape:
            raise ValueError("got weights with shape {}, expected shape {}".format(
                new_weights.shape, self._weights.shape))

        if not new_bias.shape == self._bias.shape:
            raise ValueError("got bias with shape {}, expected shape {}".format(
                new_bias.shape, self._bias.shape))

        self._weights = new_weights

    def __repr__(self):

        if self._weights is None:
            return """Fully Connected layer: type = {}, num_neurons = {}, activation = {},
                weights_shape = None, bias_shape = None""".format(
                self.type, self.num_neurons, self._activation)

        else:
            return """Fully Connected layer: type = {}, num_neurons = {}, activation = {},
                weights_shape = {}, bias_shape = {}""".format(
                self.type, self.num_neurons, self._activation,
                self._weights.shape, self._bias.shape)
