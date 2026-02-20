import os.path
import random
from typing import List
import time
import numpy as np
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt

from NNNet.net import NNNet, BASE_DIR
from concurrent.futures import ProcessPoolExecutor, as_completed
from NNNet import data_loader as dl

def parse_filename(filename):
    """
    Extrahiert Metriken aus dem Dateinamen.
    Format: nnnet_EP-{epoch}_ACC-{acc}_VALLOSS-{valloss}_VALACC-{valacc}_T-{time}.npz
    """
    pattern = r"nnnet_EP-(\d+)_ACC-([\d.]+)_VALLOSS-([\d.]+)_VALACC-([\d.]+)_T-([\d.]+)\.npz"
    match = re.match(pattern, filename)

    if match:
        return {
            'epoch': int(match.group(1)),
            'acc': float(match.group(2)),
            'valloss': float(match.group(3)),
            'valacc': float(match.group(4)),
            'time': float(match.group(5))
        }
    return None

def create_training_plot(model_dir, final_epoch):
    """
    Erstellt eine Trainings-Grafik für alle Epochen im angegebenen Ordner.

    Args:
        model_dir: Pfad zum Modell-Ordner
        final_epoch: Letzte Epoche (für den Titel)
    """
    model_dir = Path(model_dir)

    # Sammle alle Daten aus den .npz Dateien
    data = []
    for file in model_dir.glob("nnnet_EP-*.npz"):
        metrics = parse_filename(file.name)
        if metrics:
            data.append(metrics)

    if not data:
        print(f"⚠️  Keine Trainingsdaten für Plot gefunden in {model_dir}")
        return

    # Sortiere nach Epoch
    data.sort(key=lambda x: x['epoch'])

    epochs = [d['epoch'] for d in data]
    acc = [d['acc'] for d in data]
    valloss = [d['valloss'] for d in data]
    valacc = [d['valacc'] for d in data]
    time_vals = [d['time'] for d in data]

    # Erstelle Figure mit 4 Subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Titel aus Ordnerstruktur extrahieren
    model_config = str(model_dir.relative_to(model_dir.parent.parent.parent.parent.parent.parent))
    fig.suptitle(f'Training Progress: {model_config}', fontsize=14, fontweight='bold')

    # 1. Training Accuracy
    ax1.plot(epochs, acc, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Accuracy', fontsize=11)
    ax1.set_title('Training Accuracy over Epochs', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([max(0, min(acc) - 0.05), 1.0])

    # 2. Validation Loss
    ax2.plot(epochs, valloss, 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss over Epochs', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. Validation Accuracy
    ax3.plot(epochs, valacc, 'g-o', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Validation Accuracy', fontsize=11)
    ax3.set_title('Validation Accuracy over Epochs', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([max(0, min(valacc) - 0.05), 1.0])

    # 4. Training Time (kumulativ)
    ax4.plot(epochs, time_vals, 'm-o', linewidth=2, markersize=4)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Cumulative Time (seconds)', fontsize=11)
    ax4.set_title('Training Time over Epochs', fontsize=12)
    ax4.grid(True, alpha=0.3)

    # Statistiken hinzufügen
    max_acc = max(acc)
    max_valacc = max(valacc)
    min_valloss = min(valloss)
    total_time = time_vals[-1]

    textstr = f'Best Train ACC: {max_acc:.4f} | Best Val ACC: {max_valacc:.4f} | Min Val Loss: {min_valloss:.4f} | Total Time: {total_time:.1f}s'
    fig.text(0.5, 0.02, textstr, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Speichern
    output_path = model_dir / "training_progress.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Grafik erstellt: {output_path}")

def cleanup_old_models(model_dir):
    """
    Behält nur das beste Modell (höchste Val Accuracy) und löscht alle anderen .npz Dateien.

    Args:
        model_dir: Pfad zum Modell-Ordner
    """
    model_dir = Path(model_dir)

    # Sammle alle Modell-Dateien mit ihren Metriken
    models = []
    for file in model_dir.glob("nnnet_EP-*.npz"):
        metrics = parse_filename(file.name)
        if metrics:
            models.append({
                'path': file,
                'valacc': metrics['valacc'],
                'metrics': metrics
            })

    if not models:
        print(f"⚠️  Keine Modelle zum Aufräumen gefunden in {model_dir}")
        return

    # Finde das beste Modell (höchste Val Accuracy)
    best_model = max(models, key=lambda x: x['valacc'])

    # Lösche alle anderen Modelle
    deleted_count = 0
    for model in models:
        if model['path'] != best_model['path']:
            try:
                model['path'].unlink()
                deleted_count += 1
            except Exception as e:
                print(f"⚠️  Fehler beim Löschen von {model['path'].name}: {e}")

    print(f"✓ Aufgeräumt: {deleted_count} Modelle gelöscht, bestes Modell behalten:")
    print(f"  → {best_model['path'].name} (Val Acc: {best_model['valacc']:.4f})")

def save_all_epochs_data(model_dir):
    """
    Speichert alle Epochen-Daten in einer JSON-Datei, bevor die Modelle gelöscht werden.

    Args:
        model_dir: Pfad zum Modell-Ordner
    """
    model_dir = Path(model_dir)

    # Sammle alle Epochen-Daten
    all_epochs = []
    for file in sorted(model_dir.glob("nnnet_EP-*.npz")):
        metrics = parse_filename(file.name)
        if metrics:
            all_epochs.append({
                'epoch': metrics['epoch'],
                'train_accuracy': metrics['acc'],
                'val_loss': metrics['valloss'],
                'val_accuracy': metrics['valacc'],
                'cumulative_time_seconds': metrics['time'],
                'model_filename': file.name
            })

    if not all_epochs:
        print(f"⚠️  Keine Epochen-Daten zum Speichern gefunden in {model_dir}")
        return

    # Sortiere nach Epoche
    all_epochs.sort(key=lambda x: x['epoch'])

    # Zusätzliche Statistiken berechnen
    summary = {
        'total_epochs': len(all_epochs),
        'best_train_accuracy': max(e['train_accuracy'] for e in all_epochs),
        'best_val_accuracy': max(e['val_accuracy'] for e in all_epochs),
        'min_val_loss': min(e['val_loss'] for e in all_epochs),
        'total_training_time_seconds': all_epochs[-1]['cumulative_time_seconds'] if all_epochs else 0,
        'best_epoch': max(all_epochs, key=lambda x: x['val_accuracy'])['epoch']
    }

    # Datenstruktur für JSON
    data = {
        'summary': summary,
        'epochs': all_epochs,
        'config': {
            'model_path': str(model_dir.relative_to(model_dir.parent.parent.parent.parent.parent.parent))
        }
    }

    # Als JSON speichern
    output_path = model_dir / "training_history.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Epochen-Daten gespeichert: {output_path.name} ({len(all_epochs)} Epochen)")

    # Zusätzlich als CSV speichern (einfacher für Excel/Pandas)
    csv_path = model_dir / "training_history.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("epoch,train_accuracy,val_loss,val_accuracy,cumulative_time_seconds,model_filename\n")
        # Daten
        for e in all_epochs:
            f.write(f"{e['epoch']},{e['train_accuracy']},{e['val_loss']},{e['val_accuracy']},"
                   f"{e['cumulative_time_seconds']},{e['model_filename']}\n")

    print(f"✓ CSV-Export erstellt: {csv_path.name}")


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
    start = time.time()

    if use_batches:

        for epoch in range(EPOCHS):
            loss = nnnet.train_epoch_batch(train_images, train_labels, batch_size=batch_size)

            correct = sum(1 for i in range(len(train_images))
                          if np.argmax(nnnet.predict(train_images[i])) == train_labels[i])
            acc = correct / len(train_images)
            end = time.time()
            took_time = f"{end - start:.3f}"
            val_loss, val_acc = nnnet.evaluate(
                test_images,
                test_labels
            )

            save_cur_model(
                nnnet,
                round(acc, 4),
                round(val_loss, 4),
                round(val_acc, 4),
                took_time,
                hidden_layers,
                seed,
                learning_rate,
                relu,
                use_batches,
                batch_size,
                use_gpu,
                epoch,
                EPOCHS
            )
            print(f"Dauer: {end - start:.3f} Sekunden")
    else:
        for epoch in range(EPOCHS):
            correct = 0
            for i in range(len(train_images)):
                x = train_images[i]
                label = train_labels[i]
                nnnet.train(x, label)
            correct = sum(1 for i in range(len(train_images))
                          if np.argmax(nnnet.predict(train_images[i])) == train_labels[i])
            acc = correct / len(train_images)

            end = time.time()
            took_time = f"{end - start:.3f}"
            val_loss, val_acc = nnnet.evaluate(
                test_images,
                test_labels
            )

            save_cur_model(
                nnnet,
                round(acc, 4),
                round(val_loss, 4),
                round(val_acc, 4),
                took_time,
                hidden_layers,
                seed,
                learning_rate,
                relu,
                use_batches,
                batch_size,
                use_gpu,
                epoch,
                EPOCHS
            )
            print(f"Dauer: {end - start:.3f} Sekunden")

    print("Training finished")

    print("Start Testing")

    correct = 0
    for i in range(len(test_images)):
        if np.argmax(nnnet.predict(test_images[i])) == test_labels[i]:
            correct += 1
    accuracy = f"{correct / len(test_images):.4f}"
    print(f"Final Test Accuracy: {accuracy}")

def save_cur_model(
        nnnet,
        accuracy,
        val_loss,
        val_acc,
        took_time,
        hidden_layers: List[int],
        seed,
        learning_rate,
        relu,
        use_batches,
        batch_size,
        use_gpu,
        epoch,
        max_epochs,
):
    print("Saving Model...")

    # ---------- Werte runden (nur Anzeige / Dateisystem) ----------
    accuracy  = round(float(accuracy), 4)
    val_loss  = round(float(val_loss), 4)
    val_acc   = round(float(val_acc), 4)
    took_time = round(float(took_time), 3)

    # ---------- Ergebnis-Dateiname ----------
    file_name = (
        f"nnnet"
        f"_EP-{epoch + 1}"  # +1 damit Epochen bei 1 statt 0 beginnen
        f"_ACC-{accuracy}"
        f"_VALLOSS-{val_loss}"
        f"_VALACC-{val_acc}"
        f"_T-{took_time}"
        f".npz"
    )

    # ---------- Ordnerstruktur ----------
    base_dir = BASE_DIR / "MNIST_numbers" / "training_saves"

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

    # ---------- Nach letzter Epoche: Grafik erstellen und aufräumen ----------
    if epoch == max_epochs - 1:  # epoch ist 0-basiert
        print("\n" + "="*80)
        print(f"Training abgeschlossen! Erstelle Grafik und räume auf...")
        print("="*80)

        save_all_epochs_data(base_dir)  # ZUERST Daten speichern
        create_training_plot(base_dir, epoch)
        cleanup_old_models(base_dir)  # DANN Modelle löschen

        print("="*80 + "\n")


if __name__ == "__main__":
    train_NNNet(
        input_size=784,
        hidden_layers=[64,64],
        output_size=10,
        seed=42,
        learning_rate=0.1,
        pre_trained=None,
        relu=True,
        use_batches=True,
        batch_size=32,
        use_gpu=False,

        EPOCHS=40,
    )
