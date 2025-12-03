import os.path

from NNNet.net import NNNet
from train_dif_configs import train_NNNet
from concurrent.futures import ProcessPoolExecutor, as_completed
from NNNet import data_loader as dl


def run_config(cfg):
    return train_NNNet(**cfg)

def test_diffrent():
    # verschiedene Hyperparameter-Kombinationen
    configs = [
        {"hidden_size": 20, "learning_rate": 0.1, "seed": 52},
    ]

    # Anzahl Prozesse z.B. = Anzahl physischer Kerne
    max_workers = 9

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_config, cfg) for cfg in configs]

        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            print(
                f"Fertig: hs={res['hidden_size']}, lr={res['learning_rate']}, "
                f"seed={res['seed']} → acc={res['accuracy']:.4f}, file={res['path']}"
            )

    print("\nAlle Runs fertig:")
    for r in results:
        print(
            f"hs={r['hidden_size']}, lr={r['learning_rate']}, "
            f"seed={r['seed']} → acc={r['accuracy']:.4f}"
        )

def train_NNNet(
        input_size=784,
        hidden_size=30,
        output_size=10,
        seed=42,
        learning_rate=0.1,
):

    train_image_path = "data/train-images.idx3-ubyte"
    train_label_path = "data/train-labels.idx1-ubyte"

    test_image_path = "data/t10k-images.idx3-ubyte"
    test_label_path = "data/t10k-labels.idx1-ubyte"

    print("load train data")
    train_images = dl.load_images(train_image_path) / 255.0
    train_labels = dl.load_labels(train_label_path)

    print("load test data")
    test_images = dl.load_images(test_image_path) / 255.0
    test_labels = dl.load_labels(test_label_path)

    print("Create neuronal network NNNet")
    nnnet = NNNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, seed=seed, learning_rate=learning_rate)

    print("initializing training loop")
    EPOCHS = 3
    TRAIN_SAMPLES = 5000

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
            if i % 500 == 0:
                print(f"EPOCH: {epoch + 1}    ---    Training Accuracy: {correct / (i + 1):.2f}")

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
    if nnnet.seed:
        path = f"NNNet_saves/nnnet_save{counter}_acc{accuracy}_hs{nnnet.hidden_size}_lr{nnnet.learning_rate}_seed{nnnet.seed}.npz"
    else:
        path = f"NNNet_saves/nnnet_save{counter}_acc{accuracy}_hs{nnnet.hidden_size}_lr{nnnet.learning_rate}_seedXXX.npz"

    exists = True
    while exists:
        if os.path.exists(path):
            counter += 1
        else:
            if nnnet.seed:
                path = f"NNNet_saves/nnnet_save{counter}_acc{accuracy}_hs{nnnet.hidden_size}_lr{nnnet.learning_rate}_seed{nnnet.seed}.npz"
            else:
                path = f"NNNet_saves/nnnet_save{counter}_acc{accuracy}_hs{nnnet.hidden_size}_lr{nnnet.learning_rate}_seedXXX.npz"
            exists = False
    nnnet.save_NNNet(path)


