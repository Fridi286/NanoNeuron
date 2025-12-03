import random
import math

import numpy as np


class NNNet():

    # defining random weight for each loayer and ech neuron and initilaizing the neuronal network
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            seed=None,
            learning_rate = 0.01,
            pre_trained = None,
    ):

        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # random weighted inputs for hidden layer
        self.W1 = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
        self.b1 = np.zeros(hidden_size)

        # random weighted inputs for output layer
        self.W2 = np.random.uniform(-0.1, 0.1, (output_size, hidden_size))
        self.b2 = np.zeros(output_size)

        if pre_trained:
            self.load_NNNet(pre_trained)

        self.learning_rate = learning_rate

    def rand(self):
        return random.uniform(-0.1, 0.1)

    # translates big values into values between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # forward propagation
    def forward(self, x):
        # Calculates activations for each neuron in the first hidden layer
        h_raw = self.W1 @ x + self.b1
        h = self.sigmoid(h_raw)

        # Calculates activations for each neuron in the output layer
        o_raw = self.W2 @ h + self.b2
        o = self.sigmoid(o_raw)

        return h, o

    # Backpropagation
    def train(self, x, label):
        h, o = self.forward(x)

        # creating ideal goal vector, the right bit is on 1
        y = np.zeros(self.output_size)
        y[label] = 1

        #calculate error in the output layer
        error_out = (o - y) * o * (1 - o)

        #calculating the impact each hidden layer neuron has to the given output
        error_hidden = (self.W2.T @ error_out) * h * (1 - h)    # using derivation of sigmoid

        # now we need to update both weights W1 and W2 using gradient descent

        #   updating W2
        self.W2 -= self.learning_rate * np.outer(error_out, h)
        self.b2 -= self.learning_rate * error_out

        #   updating W1
        self.W1 -= self.learning_rate * np.outer(error_hidden, x)
        self.b1 -= self.learning_rate * error_hidden

    def predict(self, x):
        _, o = self.forward(x)
        # Theoretically the right number should have the highest actication
        return np.argmax(o)

    def predict_debug(self, x):
        _, o = self.forward(x)
        # Theoretically the right number should have the highest actication
        return np.argmax(o), o

    def save_NNNet(self, path):
        np.savez(path, W1=self.W1, W2=self.W2, b1=self.b1, b2=self.b2)

    def load_NNNet(self, path):
        data = np.load(path)
        self.W1 = data["W1"]
        self.W2 = data["W2"]
        self.b1 = data["b1"]
        self.b2 = data["b2"]