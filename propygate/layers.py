import numpy as np
import propygate.activations as activations


class FullyConnected:

    def __init__(self, num_neurons, activation=None, input_dim=None):

        self.num_neurons = num_neurons
        self._activation = activation if activation else "identity"
        self.input_dim = input_dim

        self.type = None
        self.weights = None
        self.bias = None

    def activation(self, x, prime=False):

        if not prime:
            activation_func = getattr(activations, self._activation)
        else:
            activation_func = getattr(activations, self._activation + "_prime")

        return activation_func(x)

    def _initialize_weights(self, previous_neurons=None):

        if self.type == "input" and previous_neurons is None:
            size = (self.input_dim, self.num_neurons)

        else:
            size = (previous_neurons, self.num_neurons)

        self.weights = np.random.uniform(-1, 1, size=size)
        self.bias = np.zeros(self.num_neurons)

    def _fprop(self, x):

        z = np.dot(x, self.weights) + self.bias

        return z, self.activation(z)

    def _bprop(self):
        pass

    def __repr__(self):

        if self.weights is None:
            return """Fully Connected layer: type = {}, num_neurons = {}, activation = {}, 
                weights_shape = None, bias_shape = None""".format(
                self.type, self.num_neurons, self.activation)

        else:
            return """Fully Connected layer: type = {}, num_neurons = {}, activation = {}, 
                weights_shape = {}, bias_shape = {}""".format(
                self.type, self.num_neurons, self.activation,
                self.weights.shape, self.bias.shape)
