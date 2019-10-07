from propygate import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


class NeuralNetwork:

    def __init__(self, neurons):
        if len(neurons) < 2:
            raise ValueError("Need atleast one input and one output layer")

        self.neurons = neurons
        self.num_weights = len(self.neurons) - 1

        self.weights = []
        self.bias = []

        self.metrics = {"train_loss": [], "train_acc": [],
                        "val_loss": [], "val_acc": []}
        self._initialized = False

    def build(self):

        for i in range(self.num_weights):
            size = (self.neurons[i+1], self.neurons[i])
            self.weights.append(np.random.uniform(-1, 1, size=size))
            self.bias.append(np.random.uniform(-1, 1, self.neurons[i+1]))

        self._initialized = True

    def forwardprop(self, inp):
        activations = [inp]
        zs = []
        for i in range(self.num_weights):
            z = np.dot(self.weights[i], activations[-1]) + self.bias[i]
            zs.append(z)
            activations.append(self.sigmoid(z))
        return zs, activations

    def backprop(self, zs, activations, y):

        gradients_w = [np.empty(w.shape) for w in self.weights]
        gradients_b = [np.empty(b.shape) for b in self.bias]

        h = 2 * (activations[-1] - y)  # MSE
        for i in reversed(range(self.num_weights)):
            h *= self.sigmoid_prime(zs[i])
            gradients_b[i] = h
            gradients_w[i] = np.outer(h, activations[i].T)
            h = np.dot(self.weights[i].T, h)

        return gradients_w, gradients_b

    def get_batches(self, x, batch_size):
        return [x[i:i+batch_size] for i in np.arange(0, len(x), batch_size)]

    def train(self, x_train, y_train, epochs=100, batch_size=64, val_data=None,
              lr=1e-2, valid_data=None):
        if len(x_train) != len(y_train):
            raise ValueError("x and y must have the same length")

        for epoch in range(epochs):
            print("Epoch {}".format(epoch + 1))

            # shuffle training data
            x_train, y_train = self.shuffle_in_unison(x_train, y_train)
            x_batched = self.get_batches(x_train, batch_size)
            y_batched = self.get_batches(y_train, batch_size)

            for i in tqdm(range(len(x_batched))):  # loop over batches
                gradients_w, gradients_b = self.get_gradients(x_batched[i], y_batched[i])
                self.stochastic_gradient_descent(gradients_w, gradients_b, lr, batch_size)

            self.evaluate(x_train, y_train, "train")

            if valid_data is not None:
                x_valid, y_valid = valid_data
                self.evaluate(x_valid, y_valid, "val")

        utils.plot_metrics(self.metrics)

    def get_gradients(self, x_batch, y_batch):
        gradients_w = [np.zeros(w.shape) for w in self.weights]
        gradients_b = [np.zeros(b.shape) for b in self.bias]

        for i in range(len(x_batch)):  # loop over one batch
            zs, activations = self.forwardprop(x_batch[i])
            d_grads_w, d_grads_b = self.backprop(zs, activations, y_batch[i])
            gradients_w = [gw + dgw for gw, dgw in zip(gradients_w, d_grads_w)]
            gradients_b = [gb + dgb for gb, dgb in zip(gradients_b, d_grads_b)]

        return gradients_w, gradients_b

    def stochastic_gradient_descent(self, gradients_w, gradients_b, lr, batch_size):
        self.weights = [w - gw * lr / batch_size for w, gw in zip(self.weights, gradients_w)]
        self.bias = [b - gb * lr / batch_size for b, gb in zip(self.bias, gradients_b)]

    def total_loss(self, outputs, y):
        return np.mean(np.sum((outputs - y) ** 2, axis=1), axis=0)  # MSE

    def accuracy(self, outputs, y):
        return np.mean(np.argmax(outputs, axis=1) == np.argmax(y, axis=1))

    def evaluate(self, x_data, y_data, prefix):
        # now has the same shape as y_data
        outputs = np.array([self.model_output(x) for x in x_data])
        self.metrics["{}_loss".format(prefix)].append(self.total_loss(outputs, y_data))
        self.metrics["{}_acc".format(prefix)].append(self.accuracy(outputs, y_data))

    @staticmethod
    def shuffle_in_unison(x, y):
        assert x.shape[0] == y.shape[0]
        perm = np.random.permutation(x.shape[0])
        return x[perm], y[perm]

    def predict(self, x):
        return self.model_output(x) / np.sum(self.model_output(x))

    def model_output(self, x):
        return self.forwardprop(x)[1][-1]

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def relu_prime(x):
        return np.heaviside(x, 0)
