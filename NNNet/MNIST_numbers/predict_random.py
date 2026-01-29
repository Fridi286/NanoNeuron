import numpy as np

from NNNet.net import NNNet
from concurrent.futures import ProcessPoolExecutor, as_completed
from NNNet import data_loader as dl
import random
import matplotlib.pyplot as plt

def load_model():
    nn = NNNet(input_size=784, hidden_size=100, seed=52, learning_rate=0.2, output_size=10)
    nn.load_NNNet("number_net_saves/5000samples/nnnet_save1_acc0.8238_hs120_lr0.2_seed52.npz")
    return nn

def show_random_test_prediction(nnnet):
    test_image_path = "data/numbers/t10k-images.idx3-ubyte"
    test_label_path = "data/numbers/t10k-labels.idx1-ubyte"

    print("load test data")
    test_images = dl.load_images(test_image_path) / 255.0
    test_labels = dl.load_labels(test_label_path)
    print(f"Test Image Anzahl: {len(test_images)}")

    # zufälliger Index
    idx = random.randint(0, 8000)

    # Bild + Label holen
    x = test_images[idx]              # shape (784,)
    label = test_labels[idx]
    img = x.reshape(28, 28)           # für Plot

    # Prediction
    pred, o = nnnet.predict_debug(x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bild links
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"True: {label} | Pred: {pred}")
    axes[0].axis("off")

    # Wahrscheinlichkeiten rechts
    axes[1].bar(np.arange(10), o)
    axes[1].set_xticks(np.arange(10))
    axes[1].set_xlabel("Digit")
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    nn = load_model()
    for i in range(10):
        show_random_test_prediction(nn)


