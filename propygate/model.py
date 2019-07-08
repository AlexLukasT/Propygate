from propygate.layers import FullyConnected
import numpy as np


class FeedForward:

    def __init__(self):

        self.num_layers = 0
        self.layers = []
        self.input_dim = None
        self.output_dim = None

        self._initialized = False

    def add(self, layer):

        if not isinstance(layer, FullyConnected):
            raise TypeError("can only add a valid layer")

        if self.num_layers == 0:
            layer.type = "input"
        else:
            layer.type = "output"

        self.layers.append(layer)
        self.num_layers += 1

        for layer in self.layers[1:-1]:
            layer.type = "hidden"

    def initialize(self):

        if self.num_layers <= 2:
            raise ValueError("atleast an input and an output layer are needed")

        if self.layers[0].input_dim is None:
            raise TypeError("need to specify input dimension for first layer")

        self.input_dim = self.layers[0].num_neurons
        self.output_dim = self.layers[-1].num_neurons

        self.layers[0]._initialize_weights()
        for i in range(self.num_layers - 1):
            previous_neurons = self.layers[i].num_neurons
            self.layers[i+1]._initialize_weights(previous_neurons=previous_neurons)

        self._initialized = True

    def train(self, x_train, y_train, n_epochs, batch_size, optimizer):

        for n in range(n_epochs):
            # ToDo: implement batch size
            for x, y in zip(x_train, y_train):
                outputs = self._forwardprop(x)
                gradients_w, gradients_b = self._backprop(outputs, y)
                optimizer.update(gradients_w, gradients_b)

    def _forwardprop(self, x):

        outputs = []
        result = x
        for layer in self.layers:
            result = layer._fprop(result)
            outputs.append(result)

        return outputs

    def _backprop(self, outputs, y):

        gradients_w = [np.empty(layer.weights.shape) for layer in self.layers]
        gradients_b = [np.empty(layer.bias.shape) for layer in self.layers]

        z_L, a_L = outputs[-1]
        h = a_L - y
        for i in np.arange(0, self.num_layers, -1):
            h *= self.layers[i].activation(z_L, prime=True)
            gradients_b[i] = h
            gradients_w[i] = np.dot(h, outputs[i-1][1])
            h = np.dot(self.layers[i].weights.T, h)

        return gradients_w, gradients_b

    def test(self, x_test, y_test):
        pass

    def __repr__(self):

        return "FeedForward model with {} layers \n".format(self.num_layers) + "\n".join(
            [str(layer) for layer in self.layers])
