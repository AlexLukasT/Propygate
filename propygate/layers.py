import numpy as np
import propygate.activations as activations


class FullyConnected:

    def __init__(self, num_neurons, activation=None, input_dim=None):

        super().__init__()

        self.num_neurons = num_neurons
        self.activation = activation
        self.input_dim = input_dim

        self.type = None
        self.weights = None
        self.bias = None

    def _initialize_weights(self, previous_neurons=None):

        if self.type == "input" and previous_neurons is None:
            size = (self.input_dim, self.num_neurons)

        else:
            size = (previous_neurons, self.num_neurons)

        self.weights = np.random.uniform(0, 1, size=size)
        self.bias = np.zeros(self.num_neurons)

    def _fprop(self, x):

        if self.activation is None:
            activation = getattr(activations, "identity")
        else:
            activation = getattr(activations, self.activation)

        return activation(np.dot(x, self.weights) + self.bias)

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
