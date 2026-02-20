from typing import List

from NNNet import net
from NNNet.MNIST_letters.data.letter_dataset_importer import CSVDataset
from NNNet.net import NNNet, BASE_DIR


def train_letters_NNNet(
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
    # Dataset laden
    dataset = CSVDataset(str(BASE_DIR / "MNIST_letters" / "data" / "emnist-letters-test.csv" / "emnist-letters-test.csv"))

    # Einzelnes Sample
    image, label = dataset.get_sample(0)

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

    TRAIN_SAMPLES = len(dataset)

    correct = 0

    # Training
    for epoch in range(EPOCHS):

        correct = 0

        for i in range(TRAIN_SAMPLES):
            image, label = dataset[i]
            nnnet.train(image, label, False)

            pred = nnnet.predict(image)
            if pred == label:
                correct += 1

        print(f"EPOCH: {epoch + 1}    ---    Training Accuracy: {correct / TRAIN_SAMPLES:.2f}")

    save_cur_model(
        nnnet,
        correct / TRAIN_SAMPLES,
        hidden_layers,
        seed,
        learning_rate,
        relu,
        use_batches,
        batch_size,
        use_gpu,
        EPOCHS,
    )

def save_cur_model(
        nnnet,
        accuracy,
        hidden_layers: List[int],
        seed,
        learning_rate,
        relu,
        use_batches,
        batch_size,
        use_gpu,
        EPOCHS,
):
    print("Saving Model...")

    # ---------- Werte runden (nur Anzeige / Dateisystem) ----------
    accuracy  = round(float(accuracy), 4)

    # ---------- Ergebnis-Dateiname ----------
    file_name = (
        f"nnnet"
        f"_EP-{EPOCHS}"
        f"_ACC-{accuracy}"
        f".npz"
    )

    # ---------- Ordnerstruktur ----------
    base_dir = BASE_DIR / "MNIST_letters" / "training_saves"

    # CPU / GPU
    base_dir /= "gpu" if use_gpu else "cpu"

    # Aktivierungsfunktion
    base_dir /= "relu" if relu else "sigmoid"

    # Batch / NoBatch
    base_dir /= "batch" if use_batches else "nobatch"

    # Batch Size (nur wenn relevant)
    if use_batches:
        base_dir /= f"BS-{batch_size}"

    # Seed / Architektur / Learning Rate
    base_dir /= f"S-{seed}"
    base_dir /= f"HL-{hidden_layers}"
    base_dir /= f"LR-{learning_rate}"

    # ---------- Ordner anlegen ----------
    base_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Kollisionen vermeiden ----------
    path = base_dir / file_name
    counter = 1
    while path.exists():
        path = base_dir / f"{file_name[:-4]}({counter}).npz"
        counter += 1

    # ---------- Speichern ----------
    nnnet.save_NNNet(str(path))

if __name__ == "__main__":
    train_letters_NNNet(
        input_size=784,
        hidden_layers=[12],
        output_size=26,
        seed=42,
        use_batches=True,
        batch_size=32,
        relu=True,
        EPOCHS=10,
    )