"""
activation_maximization_digits.py

Erzeugt ein "Bild", das ein bereits trainiertes NNNet maximal als Zielziffer klassifiziert
(Activation Maximization / Feature Visualization).

- Nutzt dein NNNet (Sigmoid Hidden + Sigmoid Output)
- Optimiert das Eingabebild x per Gradientenanstieg auf o[target]
- Optional: Regularisierung + Glättung, damit es weniger "Noise" wird
"""

from __future__ import annotations

import random
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Dein NNNet (unverändert, nur hier reinkopiert)
# -----------------------------
class NNNet:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        seed=None,
        learning_rate=0.01,
        pre_trained=None,
    ):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.uniform(-0.1, 0.1, (output_size, hidden_size))
        self.b2 = np.zeros(output_size)

        if pre_trained:
            self.load_NNNet(pre_trained)

        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # x: (784,)
        h_raw = self.W1 @ x + self.b1        # (hidden,)
        h = self.sigmoid(h_raw)              # (hidden,)

        o_raw = self.W2 @ h + self.b2        # (10,)
        o = self.sigmoid(o_raw)              # (10,)

        return h, o

    def predict_debug(self, x):
        _, o = self.forward(x)
        return np.argmax(o), o

    def save_NNNet(self, path):
        np.savez(path, W1=self.W1, W2=self.W2, b1=self.b1, b2=self.b2)

    def load_NNNet(self, path):
        data = np.load(path)
        self.W1 = data["W1"]
        self.W2 = data["W2"]
        self.b1 = data["b1"]
        self.b2 = data["b2"]


# -----------------------------
# Laden des Modells (du kannst hier deinen Pfad einsetzen)
# -----------------------------
def load_model() -> NNNet:
    nn = NNNet(input_size=784, hidden_size=120, seed=52, learning_rate=0.2, output_size=10)
    nn.load_NNNet(r"C:\Users\fridi\PycharmProjects\NanoNeuron\NNNet_saves\59999samples\nnnet_save1_acc0.9710_hs120_lr0.25_seed1000.npz")
    return nn


# -----------------------------
# Hilfen: Glättung & Visualisierung
# -----------------------------
def smooth3x3(img_2d: np.ndarray) -> np.ndarray:
    """Kleiner 3x3 Mittelwertfilter (ohne SciPy)."""
    p = np.pad(img_2d, 1, mode="edge")
    out = (
        p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:] +
        p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
        p[2:,   0:-2] + p[2:,   1:-1] + p[2:,   2:]
    ) / 9.0
    return out.astype(np.float32)


def show_result(x: np.ndarray, o: np.ndarray, target: int, title: str = "") -> None:
    img = x.reshape(28, 28)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(img, cmap="gray")
    ax1.set_title(title or f"Optimiertes Bild (target={target})")
    ax1.axis("off")

    ax2.bar(range(10), o)
    ax2.set_xlabel("Ziffer")
    ax2.set_ylabel("Aktivierung (Sigmoid)")
    ax2.set_title(f"NN-Ausgabe (target={target}: {o[target]:.4f})")
    ax2.set_xticks(range(10))
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Kern: Gradienten für o[target] bzgl. x
# -----------------------------
def grad_input_for_target(nn: NNNet, x: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Liefert:
      grad_x: Gradient von o[target] nach x (784,)
      o: Output (10,)
      h: Hidden activations (hidden,)
    """

    # Forward
    h_raw = nn.W1 @ x + nn.b1
    h = 1 / (1 + np.exp(-h_raw))

    o_raw = nn.W2 @ h + nn.b2
    o = 1 / (1 + np.exp(-o_raw))

    # d o[target] / d o_raw: nur target-Komponente ungleich 0
    # sigmoid'(z) = s(z)*(1-s(z))
    grad_o_raw = np.zeros_like(o)
    grad_o_raw[target] = o[target] * (1 - o[target])  # d o_t / d o_raw_t

    # o_raw = W2 @ h + b2  -> d o_raw / d h = W2
    # -> d o_t / d h = W2^T @ grad_o_raw
    grad_h = nn.W2.T @ grad_o_raw  # (hidden,)

    # h = sigmoid(h_raw) -> d h / d h_raw = h*(1-h)
    grad_h_raw = grad_h * (h * (1 - h))  # (hidden,)

    # h_raw = W1 @ x + b1 -> d h_raw / d x = W1
    grad_x = nn.W1.T @ grad_h_raw  # (784,)

    return grad_x.astype(np.float32), o.astype(np.float32), h.astype(np.float32)


# -----------------------------
# Optimierung: Activation Maximization
# -----------------------------
def optimize_image(
    nn: NNNet,
    target: int,
    steps: int = 2000,
    lr: float = 2.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    l2_lambda: float = 0.001,       # hält Pixel klein / verhindert extremes "Rauschen"
    mean_lambda: float = 0.05,      # bestraft zu helle Bilder (MNIST: viel Hintergrund)
    smooth_every: int = 5,          # alle N Schritte glätten
    seed: int | None = 42,
    init: str = "dark_noise",       # "dark_noise" oder "uniform"
) -> np.ndarray:
    """
    Optimiert x (784,) so, dass o[target] maximiert wird.
    """

    rng = np.random.default_rng(seed)

    # Init
    if init == "dark_noise":
        # MNIST-artig: überwiegend dunkel, leichtes Rauschen
        x = rng.normal(loc=0.0, scale=0.08, size=(784,)).astype(np.float32)
        x = np.clip(x, clip_min, clip_max)
    else:
        x = rng.uniform(clip_min, clip_max, size=(784,)).astype(np.float32)

    best_x = x.copy()
    best_score = -1e9

    for t in range(steps):
        grad_x, o, _h = grad_input_for_target(nn, x, target)

        # Regularisierung (Gradientenabzug, weil wir MAXIMIEREN wollen)
        # L2: -lambda * ||x||^2  -> grad = -2*lambda*x
        grad_x = grad_x - (2.0 * l2_lambda * x)

        # Mean penalty: -mean_lambda * mean(x)
        # mean(x) = sum(x)/784 -> grad = 1/784 pro Element
        grad_x = grad_x - (mean_lambda * (1.0 / 784.0))

        # Gradient Ascent Schritt
        x = x + lr * grad_x

        # Clip in Pixelrange
        x = np.clip(x, clip_min, clip_max)

        # Optional glätten (im 2D-Raum), dann wieder clip
        if smooth_every > 0 and (t + 1) % smooth_every == 0:
            img = x.reshape(28, 28)
            img = smooth3x3(img)
            x = np.clip(img.reshape(784,), clip_min, clip_max)

        score = float(o[target])
        if score > best_score:
            best_score = score
            best_x = x.copy()

        # Logging
        if (t + 1) % 200 == 0 or t == 0:
            pred, out = nn.predict_debug(x)
            print(f"step {t+1:4d}/{steps} | o[target]={out[target]:.4f} | pred={pred} | max={out.max():.4f}")

    return best_x


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    nn = load_model()

    target_digit = 7  # <- ändern wie du willst
    x_best = optimize_image(
        nn,
        target=target_digit,
        steps=2000,
        lr=2.0,
        l2_lambda=0.001,
        mean_lambda=0.05,
        smooth_every=5,
        init="dark_noise",
        seed=42,
    )

    pred, o = nn.predict_debug(x_best)
    print("\nFinal:")
    print("pred:", pred)
    print("out:", o)
    show_result(x_best, o, target_digit, title="Activation Maximization Ergebnis")
