import os.path
import random
from typing import List
import time

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
        use_batches=False,
        batch_size=32,
        use_gpu=False,

        EPOCHS=3,
):
    if not use_batches: batch_size = 1

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
        relu=relu,
        use_gpu=use_gpu,
    )

    print("initializing training loop")
    # GPU-Daten-Konvertierung
    if use_gpu:
        import cupy as cp
        train_images = cp.asarray(train_images)
        test_images = cp.asarray(test_images)

    print("Start Training...")

    if use_batches:
        for epoch in range(EPOCHS):
            start = time.time()
            loss = nnnet.train_epoch_batch(train_images, train_labels, batch_size=batch_size)

            correct = sum(1 for i in range(len(train_images))
                          if nnnet.predict(train_images[i]) == train_labels[i])
            acc = correct / len(train_images)
            print(f"EPOCH {epoch + 1} --- Loss: {loss:.4f} --- Train Acc: {acc:.4f}")

            end = time.time()
            print(f"Dauer: {end - start:.3f} Sekunden")
    else:
        for epoch in range(EPOCHS):
            start = time.time()
            correct = 0
            for i in range(len(train_images)):
                x = train_images[i]
                label = train_labels[i]
                nnnet.train(x, label)

                if nnnet.predict(x) == label:
                    correct += 1

            print(f"EPOCH {epoch + 1} --- Training Accuracy: {correct / len(train_images):.4f}")

            end = time.time()
            print(f"Dauer: {end - start:.3f} Sekunden")

    print("Training finished")

    print("Start Testing")

    correct = 0
    for i in range(len(test_images)):
        if nnnet.predict(test_images[i]) == test_labels[i]:
            correct += 1
    accuracy = f"{correct / len(test_images):.4f}"
    print(f"Final Test Accuracy: {accuracy}")

    print("Saving Model...")
    counter = 0
    model_details = (f""
                     f"_EP-{EPOCHS}"
                     f"_ACC-{accuracy}"
                     f"_BS-{batch_size}"
                     f"_R-{relu}"
                     f"_G-{use_gpu}"
                     f"_IS-{input_size}"
                     f"_LR-{learning_rate}"
                     f"_S-{nnnet.seed}"
                     f"_HL-{hidden_layers}"
                     )
    file_name = f"nnnet" + model_details + ".npz"
    path = str(BASE_DIR / "MNIST_numbers" / "training_saves" / file_name)

    exists = True
    while exists:
        if os.path.exists(path):
            counter += 1
            file_name = f"nnnet" + model_details + f"({counter})" + ".npz"
            path = str(BASE_DIR / "MNIST_numbers" / "training_saves" / file_name)
        else:
            exists = False
    nnnet.save_NNNet(path)


if __name__ == "__main__":
    train_NNNet(
        input_size=784,
        hidden_layers=[20],
        output_size=10,
        seed=42,
        learning_rate=0.01,
        pre_trained=None,
        relu=False,
        use_batches=True,
        batch_size=64,
        use_gpu=False,

        EPOCHS=3,
    )
