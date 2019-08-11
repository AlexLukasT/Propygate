from propygate import layers, optimizers, losses
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


class FeedForward:

    def __init__(self):

        self.num_layers = 0
        self.layers = []
        self.input_dim = None
        self.output_dim = None
        self.optimizer = None
        self.loss = None

        self.metrics = {"train_loss": [], "train_acc": [],
                        "val_loss": [], "val_acc": []}

        self._initialized = False

    def add(self, layer):

        if not isinstance(layer, layers.FullyConnected):
            raise TypeError("can only add a valid layer")

        if self.num_layers == 0:
            layer.type = "input"
        else:
            layer.type = "output"

        self.layers.append(layer)
        self.num_layers += 1

        for layer in self.layers[1:-1]:
            layer.type = "hidden"

    def initialize(self, optimizer, loss):

        if not isinstance(optimizer, optimizers.Optimizer):
            raise TypeError("pass in a valid optimizer")
        self.optimizer = optimizer

        if not isinstance(loss, losses.Loss):
            raise TypeError("pass in a valid loss")
        self.loss = loss

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

    def train(self, x_train, y_train, epochs=100, batch_size=64, val_data=None):

        n = x_train.shape[0]

        for epoch in range(epochs):
            print("Epoch {}".format(epoch + 1))
            train_loss_tmp = []
            train_acc_tmp = []
            gradients_w_tmp = []  # for storing the gradients per batch
            gradients_b_tmp = []

            x_train, y_train = self.shuffle(x_train, y_train)  # shuffle training data

            # round up to also use last incomplete batch if it exists
            n_iter = int(np.ceil(n / batch_size))
            for i in tqdm(range(n_iter)):
                iter_batch = n % batch_size if i == n_iter - 1 else batch_size
                for j in range(iter_batch):
                    activations = self._forwardprop(x_train[i * batch_size + j])
                    y = y_train[i * batch_size + j]
                    train_acc_tmp.append(np.argmax(activations[-1]) == np.argmax(y))
                    train_loss_tmp.append(np.mean(self.loss(activations[-1], y)))
                    gradients_w, gradients_b = self._backprop(activations, y)
                    gradients_w_tmp.append(gradients_w)
                    gradients_b_tmp.append(gradients_b)

                gradients_w_mean = self._gradient_sum(gradients_w_tmp)
                gradients_b_mean = self._gradient_sum(gradients_b_tmp)
                weights = self.get_weights()
                new_weights = self.optimizer.update(weights, gradients_w_mean, gradients_b_mean, batch_size)
                self.set_weights(new_weights)

            if val_data:
                x_val, y_val = val_data
                outputs = [self.model_output(x) for x in x_val]
                self.metrics["val_loss"].append(self.loss.total(outputs, y_val))
                self.metrics["val_acc"].append(self.accuracy(outputs, y_val))

            self.metrics["train_loss"].append(np.mean(train_loss_tmp))
            self.metrics["train_acc"].append(np.sum(train_acc_tmp) / n)
            print("")

    @staticmethod
    def _gradient_sum(gradients):
        mean_gradients = []
        for i in range(len(gradients[0])):  # sum over all layers
            tmp = []
            for j in range(len(gradients)):  # sum over batch size
                tmp.append(gradients[j][i])
            mean_gradients.append(np.sum(tmp, axis=0))
        return mean_gradients

    @staticmethod
    def shuffle(x, y):
        assert x.shape[0] == y.shape[0]
        perm = np.random.permutation(x.shape[0])
        return x[perm], y[perm]

    @staticmethod
    def accuracy(outputs, labels):
        return np.sum([np.argmax(a) == np.argmax(y) for (a, y) in zip(outputs, labels)]) / len(outputs)

    def plot_metrics(self, filename="metrics.png", **kwargs):
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        epochs = np.arange(1, len(self.metrics["train_loss"]) + 1)

        for metric, values in self.metrics.items():
            if metric.endswith("loss") and values:
                axes[0].plot(epochs, values, label=metric, linewidth=1.5, **kwargs)
            elif metric.endswith("acc") and values:
                axes[1].plot(epochs, values, label=metric, linewidth=1.5, **kwargs)
            else:
                continue
        axes[1].set_xlabel("Epoch", fontsize=15)
        axes[0].set_ylabel("Loss [a.u.]", fontsize=15)
        axes[1].set_ylabel("Accuracy [%]", fontsize=15)
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=15)
        fig.tight_layout()
        fig.savefig(filename, dpi=150)

    def predict(self, x):
        if len(x.shape) > 1:
            x = x.flatten()

        if x.shape[1:] == self.input_dim:
            # normalize to probability
            return [self.model_output(x_sample) / np.sum(self.model_output(x_sample)) for x_sample in x]
        else:
            return self.model_output(x) / np.sum(self.model_output(x))

    def model_output(self, x):
        return self._forwardprop(x)[-1]

    def _forwardprop(self, x):

        if len(x.shape) > 1:
            x = x.flatten()

        activations = [x]
        a = x
        for layer in self.layers:
            a = layer._fprop(a)
            activations.append(a)

        return activations

    def _backprop(self, activations, y):

        gradients_w = [np.empty(layer._weights.shape) for layer in self.layers]
        gradients_b = [np.empty(layer._bias.shape) for layer in self.layers]

        h = self.loss(activations[-1], y, prime=True)
        for i in reversed(range(self.num_layers)):
            a_l = activations[i]  # activations[0] = input
            prev_weights = self.layers[i]._weights
            h_new = self.layers[i]._bprop(h, a_l, prev_weights)
            h = h_new
            gradients_w[i] = self.layers[i].gradient_w
            gradients_b[i] = self.layers[i].gradient_b

        return gradients_w, gradients_b

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weight, bias = layer.get_weights()
            weights.append([weight, bias])
        return weights

    def set_weights(self, weights):
        for i, (new_weights, new_bias) in enumerate(weights):
            self.layers[i].set_weights(new_weights, new_bias)

    def test(self, x_test, y_test):
        pass

    def __repr__(self):

        return "FeedForward model with {} layers \n".format(self.num_layers) + "\n".join(
            [str(layer) for layer in self.layers])
