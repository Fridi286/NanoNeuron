import os.path
import random
from typing import List

from NNNet.net import NNNet, BASE_DIR
from concurrent.futures import ProcessPoolExecutor, as_completed
from NNNet import data_loader as dl

def train_NNNet(
        input_size,
        hidden_layers: List[int],
        output_size,
        seed=None,
        learning_rate=0.01,
        pre_trained=None,
        relu=None,
):

    train_image_path = BASE_DIR / "MNIST_numbers" / "data" / "train-images.idx3-ubyte"
    train_label_path = BASE_DIR / "MNIST_numbers" / "data" / "train-labels.idx1-ubyte"

    test_image_path = BASE_DIR / "MNIST_numbers" / "data" / "t10k-images.idx3-ubyte"
    test_label_path = BASE_DIR / "MNIST_numbers" / "data" / "t10k-labels.idx1-ubyte"

    print("load train data")
    train_images = dl.load_images(train_image_path) / 255.0
    train_labels = dl.load_labels(train_label_path)

    print("load test data")
    test_images = dl.load_images(test_image_path) / 255.0
    test_labels = dl.load_labels(test_label_path)

    print(len(train_images))

    print(f"Create neuronal network NNNet with configuration:\n"
          f"Input Size: {input_size}\n"
          f"Hidden Layers: {str(hidden_layers)}\n"
          f"Output Size: {output_size}\n"
          f"Seed: {seed}\n"
          f"Learning Rate: {learning_rate}\n"
          f"Pre-Trained?: {pre_trained}\n")
    nnnet = NNNet(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        seed=seed,
        learning_rate=learning_rate,
        pre_trained=pre_trained,
    )

    print("initializing training loop")
    EPOCHS = 5
    TRAIN_SAMPLES = len(train_images)

    print("Start Training...")

    for epoch in range(EPOCHS):
        correct = 0

        for i in range(TRAIN_SAMPLES):
            x = train_images[i]
            label = train_labels[i]

            nnnet.train(x, label)

            # Testing testwise while running
            pred = nnnet.predict(x)
            if pred == label:
                correct += 1

            # every 500 samples output current accuracy
            #if i % 500 == 0:
            #    print(f"EPOCH: {epoch + 1}    ---    Training Accuracy: {correct / (i + 1):.2f}")

        # accuracry after each epoch
        print(f"EPOCH: {epoch + 1}    ---    Training Accuracy: {correct / TRAIN_SAMPLES:.2f}")

    print("Training finished")

    print("Start Testing")

    correct = 0
    for i in range(len(test_images)):
        if nnnet.predict(test_images[i]) == test_labels[i]:
            correct += 1
    accuracy = f"{correct / len(test_images):.4f}"
    print(f"Final Test Accuracy: {accuracy}")


    # ------------------ Save to File -------------------------

    counter = 1
    path = str(BASE_DIR / "MNIST_numbers" / "training_saves" / f"nnnet_number{counter}_ACC{accuracy}_")

    exists = True
    while exists:
        if os.path.exists(path):
            counter += 1
            path = str(BASE_DIR / "MNIST_numbers" / "training_saves" / f"nnnet_number_{counter}_ACC{accuracy}_")
        else:
            exists = False
    nnnet.save_NNNet(path)

if __name__ == "__main__":
    train_NNNet(
        input_size=784,
        hidden_layers=[512, 256, 128],
        output_size=10,
        seed=None,
        learning_rate=0.15,
        pre_trained=None,
        relu=True,
    )
