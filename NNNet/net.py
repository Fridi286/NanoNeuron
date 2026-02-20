import random
from typing import List

import random
from typing import List
from pathlib import Path

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU (CuPy) verfügbar")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU nicht verfügbar, nutze CPU (NumPy)")

import numpy as np

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class NNNet:
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
            use_gpu=False,
    ):

        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np  # NumPy oder CuPy wählen

        self.z = None
        self.a = None
        if seed is None:
            seed = random.randint(0, 2 ** 32 - 1)

        self.seed = seed
        np.random.seed(seed)
        if self.use_gpu:
            cp.random.seed(seed)
        random.seed(seed)

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.use_relu= relu
        self.use_gpu = use_gpu

        self.W = []
        self.b = []
        prev_size = input_size

        #Random weights for each hidden layer
        for hidden_size in hidden_layers:
            self.W.append(self.xp.random.uniform(-0.1, 0.1, (hidden_size, prev_size)))
            self.b.append(self.xp.zeros(hidden_size))
            prev_size = hidden_size

        # Output-Layer hinzufügen
        W = np.random.uniform(-0.1, 0.1, (output_size, prev_size))
        self.W.append(self.xp.asarray(W) if self.use_gpu else W)
        self.b.append(self.xp.zeros(output_size))

        if pre_trained:
            self.load_NNNet(pre_trained)

        self.learning_rate = learning_rate

    # =======================FORWARDING + BACKPROPAGATION==========================

    # -----------------------------------------------------------------------------
    # FORWARDING AND BACKPORPAGATON WITH SIGMOID
    # -----------------------------------------------------------------------------
    def sigmoid(self, x):
        return 1 / (1 + self.xp.exp(-x))

    # forward propagation
    def forward(self, x):
        if self.use_relu: return self.forward_relu(x)
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
        if self.use_relu: return self.train_relu(x, label, letter=False)
        o = self.forward(x)

        y = self.xp.zeros(self.output_size)
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

            dW = self.xp.outer(delta, a_prev)
            db = delta

            self.W[l] -= self.learning_rate * dW
            self.b[l] -= self.learning_rate * db

            if l > 0:
                delta = delta_prev

        return o

    #==============================================================================
    #      FORWARDING AND BACKPORPAGATON WITH RELU AND SOFTMAX - W/O Batch
    #==============================================================================
    def softmax(self, z):
        z = z - self.xp.max(z, axis=-1, keepdims=True)
        exp_z = self.xp.exp(z)
        return exp_z / self.xp.sum(exp_z, axis=-1, keepdims=True)

    def relu(self, x):
        return self.xp.maximum(0, x)

    # forward propagation
    def forward_relu(self, x):
        self.a = [x]  # Activations, a[0] = input
        self.z = []  # Preactivations

        for l in range(len(self.W)):
            z = self.W[l] @ self.a[l] + self.b[l]
            self.z.append(z)

            if l == len(self.W) - 1:
                a = self.softmax(z)  # Output-Layer
            else:
                a = self.relu(z)  # Hidden-Layer

            self.a.append(a)

        return self.a[-1]  # Output

    # Backpropagation with relu and softmax
    def train_relu(self, x, label, letter):
        o = self.forward_relu(x)

        y = self.xp.zeros(self.output_size)  # self.xp statt np
        y[label - 1 if letter else label] = 1

        delta = o - y

        for l in range(len(self.W) - 1, -1, -1):
            a_prev = self.a[l]

            if l > 0:
                relu_grad = (self.z[l - 1] > 0).astype(float)
                delta_prev = (self.W[l].T @ delta) * relu_grad

            dW = self.xp.outer(delta, a_prev)  # self.xp
            db = delta

            self.W[l] -= self.learning_rate * dW
            self.b[l] -= self.learning_rate * db

            if l > 0:
                delta = delta_prev

        return o
    #==============================================================================
    #                               BATCH TRAINING
    #==============================================================================
    def train_relu_batch(self, x_batch, labels_batch, letter=False):
        is_single = x_batch.ndim == 1
        if is_single:
            x_batch = x_batch.reshape(1, -1)
            labels_batch = self.xp.array([labels_batch])  # self.xp

        batch_size = x_batch.shape[0]

        # Daten auf GPU verschieben
        if self.use_gpu and not isinstance(x_batch, cp.ndarray):
            x_batch = cp.asarray(x_batch)
            labels_batch = cp.asarray(labels_batch)

        self.a = [x_batch]
        self.z = []

        for l in range(len(self.W)):
            z = (self.W[l] @ self.a[l].T).T + self.b[l]
            self.z.append(z)

            if l == len(self.W) - 1:
                z_shifted = z - self.xp.max(z, axis=1, keepdims=True)  # self.xp
                exp_z = self.xp.exp(z_shifted)
                a = exp_z / self.xp.sum(exp_z, axis=1, keepdims=True)
            else:
                a = self.xp.maximum(0, z)  # self.xp

            self.a.append(a)

        o = self.a[-1]

        y = self.xp.zeros((batch_size, self.output_size))  # self.xp
        for i, label in enumerate(labels_batch):
            label_val = int(label) if self.use_gpu else label
            y[i, label_val - 1 if letter else label_val] = 1

        delta = o - y

        for l in range(len(self.W) - 1, -1, -1):
            a_prev = self.a[l]

            dW = (delta.T @ a_prev) / batch_size
            db = self.xp.mean(delta, axis=0)  # self.xp

            if l > 0:
                relu_grad = (self.z[l - 1] > 0).astype(float)
                delta = (delta @ self.W[l]) * relu_grad

            self.W[l] -= self.learning_rate * dW
            self.b[l] -= self.learning_rate * db

        return o[0] if is_single else o

    def train_epoch_batch(self, x_train, y_train, batch_size=32, letter=False):
        # Daten auf GPU verschieben
        if self.use_gpu:
            if not isinstance(x_train, cp.ndarray):
                x_train = cp.asarray(x_train)
            if not isinstance(y_train, cp.ndarray):
                y_train = cp.asarray(y_train)

        n_samples = x_train.shape[0]
        indices = self.xp.arange(n_samples)  # self.xp
        self.xp.random.shuffle(indices)

        total_loss = 0
        n_batches = 0

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            output = self.train_relu_batch(x_batch, y_batch, letter)

            y_one_hot = self.xp.zeros((len(y_batch), self.output_size))  # self.xp
            for i, label in enumerate(y_batch):
                label_val = int(label) if self.use_gpu else label
                y_one_hot[i, label_val - 1 if letter else label_val] = 1

            loss = -self.xp.sum(y_one_hot * self.xp.log(output + 1e-8)) / len(y_batch)  # self.xp
            total_loss += float(loss) if self.use_gpu else loss
            n_batches += 1

        return total_loss / n_batches

    # ===========================================================================
    # Predict
    # ===========================================================================

    def predict(self, x):
        # Auf GPU verschieben falls nötig
        if self.use_gpu and not isinstance(x, cp.ndarray):
            x = cp.asarray(x)

        # Forward pass
        if self.use_relu:
            o = self.forward_relu(x)
        else:
            o = self.forward(x)

        # Zurück auf CPU für Ausgabe
        if self.use_gpu:
            o = cp.asnumpy(o)

        return o


    # ===========================================================================
    # Evaluate
    # ===========================================================================

    def evaluate(self, x_val, y_val, batch_size=256, letter=False):
        # ggf. auf GPU schieben
        if self.use_gpu:
            if not isinstance(x_val, cp.ndarray):
                x_val = cp.asarray(x_val)
            if not isinstance(y_val, cp.ndarray):
                y_val = cp.asarray(y_val)

        n = x_val.shape[0]
        total_loss = 0.0
        total_correct = 0

        eps = 1e-8

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = x_val[start:end]
            yb = y_val[start:end]

            # ---- forward (ohne Updates) ----
            if self.use_relu:
                # Batch-forward über deine Batch-Forward-Logik (ohne Backprop)
                # Wir nutzen hier die gleiche Mathe wie in train_relu_batch, nur ohne Updates:
                a = xb
                z_list = []

                for l in range(len(self.W)):
                    z = (self.W[l] @ a.T).T + self.b[l]
                    z_list.append(z)

                    if l == len(self.W) - 1:
                        z_shifted = z - self.xp.max(z, axis=1, keepdims=True)
                        exp_z = self.xp.exp(z_shifted)
                        a = exp_z / self.xp.sum(exp_z, axis=1, keepdims=True)
                    else:
                        a = self.xp.maximum(0, z)

                o = a  # (B, C) Softmax-Wahrscheinlichkeiten

                # ---- Loss: -log(p_true) gemittelt ----
                # labels evtl. int cast bei GPU
                if self.use_gpu:
                    yb_idx = yb.astype(self.xp.int32)
                else:
                    yb_idx = yb

                if letter:
                    yb_idx = yb_idx - 1

                p_true = o[self.xp.arange(o.shape[0]), yb_idx]
                p_true = self.xp.clip(p_true, eps, 1.0)
                batch_loss = -self.xp.mean(self.xp.log(p_true))

                # ---- Accuracy ----
                preds = self.xp.argmax(o, axis=1)
                batch_correct = self.xp.sum(preds == yb_idx)

            else:
                # Sigmoid-Forward ist bei dir nur für EIN sample sauber implementiert.
                # Für Val reicht: pro sample forward, dann MSE gegen One-Hot.
                batch_loss = 0.0
                batch_correct = 0
                for i in range(xb.shape[0]):
                    o = self.forward(xb[i])  # (C,)

                    # One-Hot
                    y = self.xp.zeros(self.output_size)
                    lbl = int(yb[i]) if self.use_gpu else yb[i]
                    y[lbl - 1 if letter else lbl] = 1

                    # MSE (wie dein Sigmoid-Training stilistisch dazu passt)
                    batch_loss += float(self.xp.mean((o - y) ** 2))

                    pred = int(self.xp.argmax(o))
                    true = int(lbl - 1 if letter else lbl)
                    batch_correct += (pred == true)

                batch_loss /= xb.shape[0]

            # ---- aufsummieren----
            total_loss += float(batch_loss) * (end - start)
            total_correct += int(batch_correct) if self.use_gpu else int(batch_correct)

        val_loss = total_loss / n
        val_acc = total_correct / n
        return val_loss, val_acc

    # ===========================================================================
    # SAVE + LOAD
    # ===========================================================================

    def save_NNNet(self, path: str):
        hidden_layers_String = f"HL"
        for i in self.hidden_layers:
            hidden_layers_String += f"-{i}"

        # Gewichte zurück auf CPU
        W_cpu = [cp.asnumpy(w) if self.use_gpu else w for w in self.W]
        b_cpu = [cp.asnumpy(b) if self.use_gpu else b for b in self.b]

        np.savez(
            path,
            W=np.array(W_cpu, dtype=object),
            b=np.array(b_cpu, dtype=object),
            input_size=self.input_size,
            hidden_layers=np.array(self.hidden_layers, dtype=int),
            output_size=self.output_size,
            learning_rate=self.learning_rate,
            seed=-1 if self.seed is None else int(self.seed),
        )

    def load_NNNet(self, path: str):
        data = np.load(path, allow_pickle=True)

        self.W = [self.xp.asarray(w) if self.use_gpu else w for w in data["W"]]
        self.b = [self.xp.asarray(b) if self.use_gpu else b for b in data["b"]]

        self.input_size = int(data["input_size"])
        self.hidden_layers = list(data["hidden_layers"].astype(int))
        self.output_size = int(data["output_size"])
        self.learning_rate = float(data["learning_rate"])

        seed = int(data["seed"])
        self.seed = None if seed == -1 else seed

