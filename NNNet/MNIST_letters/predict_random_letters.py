import numpy as np
from NNNet.net import NNNet
from training_data.letters.csv_dataset import CSVDataset
import random
import matplotlib.pyplot as plt


def load_model():
    nn = NNNet(input_size=784, hidden_size=100, output_size=26, seed=42, learning_rate=0.1)
    nn.load_NNNet("C:\\Users\\fridi\\PycharmProjects\\NanoNeuron\\NNNet_saves_ExtendedMNIST\\Letters\\FirstLetterNet.npz")
    return nn


def show_random_train_prediction(nnnet, dataset):
    # Zufälliger Index aus dem Trainingsdatensatz
    idx = random.randint(0, len(dataset) - 1)

    # Bild + Label holen
    x, label = dataset[idx]
    img = x.reshape(28, 28)

    # Prediction
    pred, o = nnnet.predict_debug(x)

    # Buchstaben A-Z
    LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bild links
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"True: {LETTERS[label]} | Pred: {LETTERS[pred]}")
    axes[0].axis("off")

    # Wahrscheinlichkeiten rechts
    axes[1].bar(np.arange(26), o)
    axes[1].set_xticks(np.arange(26))
    axes[1].set_xticklabels(LETTERS, fontsize=8)
    axes[1].set_xlabel("Letter")
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Trainingsdatensatz laden
    dataset = CSVDataset(
        'C:\\Users\\fridi\\PycharmProjects\\NanoNeuron\\data\\letters\\emnist-letters-train.csv\\emnist-letters-train.csv')

    # Modell laden
    nn = load_model()

    # 10 zufällige Predictions anzeigen
    for i in range(10):
        show_random_train_prediction(nn, dataset)
