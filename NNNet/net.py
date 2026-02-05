import random
import math
from typing import List

import numpy as np
from pydantic import BaseModel

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class NNNet():

    # defining random weight for each loayer and ech neuron and initilaizing the neuronal network
    def __init__(
            self,
            input_size,
            hidden_layers: List[int],
            output_size,
            seed=None,
            learning_rate = 0.01,
            pre_trained = None,
            relu=False,
    ):

        self.z = None
        self.a = None
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.relu= relu

        self.W = []
        self.b = []
        prev_size = input_size

        #Random weights for each hidden layer
        for hidden_size in hidden_layers:
            self.W.append(
                np.random.uniform(-0.1, 0.1, (hidden_size, prev_size))
            )
            self.b.append(
                np.zeros(hidden_size)
            )
            prev_size = hidden_size

        # Output-Layer hinzufügen
        self.W.append(
            np.random.uniform(-0.1, 0.1, (output_size, prev_size))
        )
        self.b.append(
            np.zeros(output_size)
        )

        if pre_trained:
            self.load_NNNet(pre_trained)

        self.learning_rate = learning_rate

    # =======================FORWARDING + BACKPROPAGATION==========================

    # -----------------------------------------------------------------------------
    # FORWARDING AND BACKPORPAGATON WITH SIGMOID
    # -----------------------------------------------------------------------------
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # forward propagation
    def forward(self, x):
        if self.relu: return self.forward_relu(self, x)
        self.a = [x]  # Activations, a[0] = input
        self.z = []  # Pre-activations

        for l in range(len(self.W)):  # W enthält ALLE Layer, inkl. Output
            z = self.W[l] @ self.a[l] + self.b[l]
            self.z.append(z)
            a = self.sigmoid(z)
            self.a.append(a)

        return self.a[-1]  # Output

    # Backpropagation
    def train(self, x, label, letter=False):
        if self.relu: return self.train_relu(self, x, label, letter=False)
        o = self.forward(x)

        y = np.zeros(self.output_size)
        y[label - 1 if letter else label] = 1

        # delta im Output-Layer
        delta = (o - y) * o * (1 - o)

        # Rückwärts durch alle Layer
        for l in range(len(self.W) - 1, -1, -1):
            a_prev = self.a[l]

            # delta fürs vorherige Layer VOR dem Update berechnen
            if l > 0:
                a_prev_act = self.a[l]
                delta_prev = (self.W[l].T @ delta) * (a_prev_act * (1 - a_prev_act))

            dW = np.outer(delta, a_prev)
            db = delta

            self.W[l] -= self.learning_rate * dW
            self.b[l] -= self.learning_rate * db

            if l > 0:
                delta = delta_prev

        return o

    # -----------------------------------------------------------------------------------
    # FORWARDING AND BACKPORPAGATON WITH RELU AND SOFTMAX
    # -----------------------------------------------------------------------------------
    def softmax(self, z):
        z = z - np.max(z)  # wichtig für Stabilität
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    # translates big values into values between 0 and infinity
    def relu_deriv(x):
        return (x > 0).astype(float)

    # forward propagation
    def forward_relu(self, x):
        self.a = [x]  # Activations, a[0] = input
        self.z = []  # Pre-activations

        for l in range(len(self.W)):
            z = self.W[l] @ self.a[l] + self.b[l]
            self.z.append(z)

            if l == len(self.W) - 1:
                a = self.softmax(z)  # Output-Layer
            else:
                a = self.relu_deriv(z)  # Hidden-Layer

            self.a.append(a)

        return self.a[-1]  # Output

    # Backpropagation with relu and softmax
    def train_relu(self, x, label, letter=False):
        o = self.forward(x)

        y = np.zeros(self.output_size)
        y[label - 1 if letter else label] = 1

        # delta im Output-Layer
        delta = o - y

        # Rückwärts durch alle Layer
        for l in range(len(self.W) - 1, -1, -1):
            a_prev = self.a[l]

            # delta fürs vorherige Layer VOR dem Update berechnen
            if l > 0:
                a_prev_act = self.a[l]
                delta_prev = (self.W[l].T @ delta) * (a_prev_act * (1 - a_prev_act))

            dW = np.outer(delta, a_prev)
            db = delta

            self.W[l] -= self.learning_rate * dW
            self.b[l] -= self.learning_rate * db

            if l > 0:
                delta = delta_prev

        return o

    # ===========================================================================
    # Predict
    # ===========================================================================

    def predict(self, x):
        o = self.forward(x)
        # Theoretically the right number should have the highest actication
        return np.argmax(o)

    def predict_debug(self, x):
        o = self.forward(x)
        # Theoretically the right number should have the highest actication
        return np.argmax(o), o

    # ===========================================================================
    # SAVE + LOAD
    # ===========================================================================

    def save_NNNet(self, path: str):
        hidden_layers_String = f"HL"
        for i in self.hidden_layers:
            hidden_layers_String += f"-{i}"
        # Speichert W und b als Listen in einer .npz
        np.savez(
            path+f"IS{self.input_size}_LR{self.learning_rate}_SEED{self.seed}_{hidden_layers_String}.npz",
            W=np.array(self.W, dtype=object),
            b=np.array(self.b, dtype=object),
            input_size=self.input_size,
            hidden_layers=np.array(self.hidden_layers, dtype=int),
            output_size=self.output_size,
            learning_rate=self.learning_rate,
            seed=-1 if self.seed is None else int(self.seed),
        )

    def load_NNNet(self, path: str):
        data = np.load(path, allow_pickle=True)

        self.W = list(data["W"])
        self.b = list(data["b"])

        # optional: Meta wiederherstellen (falls du willst)
        self.input_size = int(data["input_size"])
        self.hidden_layers = list(data["hidden_layers"].astype(int))
        self.output_size = int(data["output_size"])
        self.learning_rate = float(data["learning_rate"])

        seed = int(data["seed"])
        self.seed = None if seed == -1 else seed

