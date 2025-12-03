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