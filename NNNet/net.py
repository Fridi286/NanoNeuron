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
        random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # random weighted inputs for hidden layer
        self.W1 = [[self.rand() for _ in range(self.input_size)]
                   for _ in range(self.hidden_size)]
        self.b1 = [0.0 for _ in range(self.hidden_size)]    #biases for w1

        # random weighted inputs for output layer
        self.W2 = [[self.rand() for _ in range(self.hidden_size)]
                   for _ in range(self.output_size)]
        self.b2 = [0.0 for _ in range(self.output_size)]    #biases for w2

        if pre_trained:
            self.load_NNNet(pre_trained)

        self.learning_rate = learning_rate

    def rand(self):
        return random.uniform(-0.1, 0.1)

    # translates big values into values between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # calculates the weighted sum of all input weights
    def dot(self, a, b):
        return sum(x * y for x, y in zip(a, b))

    # forward propagation
    def forward(self, x):
        # Calculates activations for each neuron in the first hidden layer
        h = [
            self.sigmoid(self.dot(self.W1[i], x) + self.b1[i])
            for i in range(self.hidden_size)
        ]
        # Calculates activations for each neuron in the output layer
        o = [
            self.sigmoid(self.dot(self.W2[i], h) + self.b2[i])
            for i in range(self.output_size)
        ]
        return h, o

    # Backpropagation
    def train(self, x, label):
        h, o = self.forward(x)

        # creating ideal goal vector, the right bit is on 1
        y = [0] * self.output_size
        y[label] = 1

        #calculate error in the output layer
        # return a list of 10 values which list the error for every output neuron
        error_out = [
            (o[i] - y[i]) * o[i] * (1 - o[i]) for i in range(self.output_size)      # using derivation of sigmoid
        ]

        #calculating the impact each hidden layer neuron has to the given output
        error_hidden = []
        for i in range(self.hidden_size):
            s = sum(error_out[j] * self.W2[j][i] for j in range(self.output_size))  # sum of all errors of all output neurons, in relation t the weight
            error_hidden.append(s * h[i] * (1-h[i]))                                  # using derivation of sigmoid

        # now we need to update both weights W1 and W2 using gradient descent

        #   updating W2
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                self.W2[i][j] -= self.learning_rate * error_out[i] * h[j]
            self.b2[i] -= self.learning_rate * error_out[i]

        #   updating W1
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                self.W1[i][j] -= self.learning_rate * error_hidden[i] * x[j]
            self.b1[i] -= self.learning_rate * error_hidden[i]

    def predict(self, x):
        _, o = self.forward(x)
        # Theoretically the right number should have the highest actication
        return max(range(self.output_size), key=lambda i: o[i])

    def save_NNNet(self, path):
        np.savez(path, W1=self.W1, W2=self.W2, b1=self.b1, b2=self.b2)

    def load_NNNet(self, path):
        data = np.load(path)
        self.W1 = data["W1"]
        self.W2 = data["W2"]
        self.b1 = data["b1"]
        self.b2 = data["b2"]