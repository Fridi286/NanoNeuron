import random
import math

def rand():
    return random.uniform(-0.1, 0.1)


class NNNEt():

    # defining random weight for each loayer and ech neuron and initilaizing the neuronal network
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # random weighted inputs for hidden layer
        self.W1 = [[rand() for _ in range(self.input_size)]
                   for _ in range(self.hidden_size)]
        self.b1 = [0.0 for _ in range(self.hidden_size)]

        # random weighted inputs for output layer
        self.W2 = [[rand() for _ in range(self.hidden_size)]
                   for _ in range(self.output_size)]
        self.b2 = [0.0 for _ in range(self.output_size)]

        self.learning_rate = 0.1

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

    def train(self, x, label):
        print("missing atm")

    def predict(self, x):
        _, o = self.forward(x)
        # Theoretically the right number should have the highest actication
        return max(range(self.output_size))

