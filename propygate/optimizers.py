import numpy as np
from propygate import utils
from tqdm import tqdm


class Optimizer:
    def __init__(self, model, learning_rate, batch_size):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.metrics = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def _apply_gradients(self, *args, **kwargs):
        raise NotImplementedError

    def get_batches(self, x, batch_size):
        return [x[i : i + batch_size] for i in np.arange(0, len(x), batch_size)]

    def train(
        self, x_train, y_train, epochs=100, val_data=None, lr=1e-2, valid_data=None,
    ):
        if len(x_train) != len(y_train):
            raise ValueError("x and y must have the same length")

        for epoch in range(epochs):
            print("Epoch {}".format(epoch + 1))

            # shuffle training data
            x_train, y_train = utils.shuffle_in_unison(x_train, y_train)
            x_batched = self.get_batches(x_train, self.batch_size)
            y_batched = self.get_batches(y_train, self.batch_size)

            for i in tqdm(range(len(x_batched))):  # loop over batches
                gradients_w, gradients_b = self.model._get_gradients(
                    x_batched[i], y_batched[i]
                )
                self._apply_gradients(gradients_w, gradients_b)

            self.evaluate(x_train, y_train, "train")

            if valid_data is not None:
                x_valid, y_valid = valid_data
                self.evaluate(x_valid, y_valid, "val")

        utils.plot_metrics(self.metrics)

    def accuracy(self, outputs, y):
        return np.mean(np.argmax(outputs, axis=1) == np.argmax(y, axis=1))

    def evaluate(self, x_data, y_data, prefix):
        # now has the same shape as y_data
        outputs = np.array([self.model.output(x) for x in x_data])
        self.metrics["{}_loss".format(prefix)].append(
            self.model.loss.total(outputs, y_data)
        )
        self.metrics["{}_acc".format(prefix)].append(self.accuracy(outputs, y_data))


class GradientDescent(Optimizer):
    def __init__(self, model, learning_rate, batch_size):
        super(GradientDescent, self).__init__(model, learning_rate, batch_size)

    def _apply_gradients(self, gradients_w, gradients_b):
        # self.model.weights = [
        #     w - gw * self.learning_rate / self.batch_size
        #     for w, gw in zip(self.model.weights, gradients_w)
        # ]
        # self.model.bias = [
        #     b - gb * self.learning_rate / self.batch_size
        #     for b, gb in zip(self.model.bias, gradients_b)
        # ]
        for i in range(len(self.model.layers)):
            self.model.layers[i].weights -= (
                gradients_w[i] * self.learning_rate / self.batch_size
            )
            self.model.layers[i].bias -= (
                gradients_b[i] * self.learning_rate / self.batch_size
            )
