"""
Skript zur Visualisierung des Trainingsverlaufs für alle trainierten MNIST-Modelle.
Liest die training_history.json Dateien und erstellt Graphen mit ACC, VALLoss, VALACC und Zeit.
"""

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Basis-Verzeichnis
BASE_DIR = Path(__file__).resolve().parent / "training_saves"
OUTPUT_DIR = Path(__file__).resolve().parent / "training_analysis"

def load_training_history(json_path):
    """
    Lädt die training_history.json Datei und gibt die Epochen-Daten zurück.

    Args:
        json_path: Pfad zur training_history.json Datei

    Returns:
        Liste von Dicts mit Epochen-Daten oder None bei Fehler
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Konvertiere das Format von training_history.json zu unserem internen Format
        epochs_data = []
        for epoch_info in data.get('epochs', []):
            epochs_data.append({
                'epoch': epoch_info['epoch'],
                'acc': epoch_info['train_accuracy'],
                'valloss': epoch_info['val_loss'],
                'valacc': epoch_info['val_accuracy'],
                'time': epoch_info['cumulative_time_seconds']
            })

        return epochs_data
    except Exception as e:
        print(f"⚠️  Fehler beim Laden von {json_path}: {e}")
        return None

def collect_model_data(min_valacc=0.0):
    """
    Durchsucht alle Unterverzeichnisse nach training_history.json Dateien.
    Gruppiert nach Modellkonfiguration (Pfad ohne training_saves).

    Args:
        min_valacc: Minimale Validation Accuracy - Modelle darunter werden aussortiert
    """
    model_data = {}

    for root, dirs, files in os.walk(BASE_DIR):
        # Suche nach training_history.json Dateien
        if 'training_history.json' in files:
            json_path = Path(root) / 'training_history.json'

            # Lade die Epochen-Daten
            epochs_data = load_training_history(json_path)

            if epochs_data:
                # Relativer Pfad als Modell-ID
                rel_path = Path(root).relative_to(BASE_DIR)
                model_id = str(rel_path)

                # Bereits sortiert nach Epoche (kommt aus JSON)
                model_data[model_id] = epochs_data

    # Filtere schlechte Modelle aus (basierend auf bester Validation Accuracy)
    if min_valacc > 0:
        filtered_data = {}
        for model_id, data in model_data.items():
            if data:
                best_valacc = max([d['valacc'] for d in data])
                if best_valacc >= min_valacc:
                    filtered_data[model_id] = data
        return filtered_data

    return model_data

def create_model_plot(model_id, data, output_dir):
    """
    Erstellt einen 4-Panel-Plot für ein Modell mit allen Metriken über die Epochen.
    """
    if not data:
        return

    epochs = [d['epoch'] for d in data]
    acc = [d['acc'] for d in data]
    valloss = [d['valloss'] for d in data]
    valacc = [d['valacc'] for d in data]
    time = [d['time'] for d in data]

    # Erstelle Figure mit 4 Subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Progress: {model_id}', fontsize=16, fontweight='bold')

    # 1. Training Accuracy
    ax1.plot(epochs, acc, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Accuracy', fontsize=12)
    ax1.set_title('Training Accuracy over Epochs', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(acc) - 0.02, 1.0])

    # 2. Validation Loss
    ax2.plot(epochs, valloss, 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss over Epochs', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 3. Validation Accuracy
    ax3.plot(epochs, valacc, 'g-o', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Validation Accuracy', fontsize=12)
    ax3.set_title('Validation Accuracy over Epochs', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([min(valacc) - 0.02, 1.0])

    # 4. Training Time (kumulativ)
    ax4.plot(epochs, time, 'm-o', linewidth=2, markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Cumulative Time (seconds)', fontsize=12)
    ax4.set_title('Training Time over Epochs', fontsize=14)
    ax4.grid(True, alpha=0.3)

    # Zusätzliche Statistiken in den Titel einfügen
    max_acc = max(acc)
    max_valacc = max(valacc)
    min_valloss = min(valloss)
    total_time = time[-1]

    textstr = f'Best Train ACC: {max_acc:.4f} | Best Val ACC: {max_valacc:.4f} | Min Val Loss: {min_valloss:.4f} | Total Time: {total_time:.1f}s'
    fig.text(0.5, 0.02, textstr, ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Speichern
    model_output_dir = output_dir / model_id
    model_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_output_dir / "training_progress.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Erstellt: {output_path}")

def create_comparison_plot(model_data, output_dir):
    """
    Erstellt einen Vergleichsplot für alle Modelle (Validation Accuracy).
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    for model_id, data in sorted(model_data.items()):
        if not data:
            continue
        epochs = [d['epoch'] for d in data]
        valacc = [d['valacc'] for d in data]

        # Kürze den Label für bessere Lesbarkeit
        label = model_id.replace('cpu/', '').replace('relu/', 'ReLU/').replace('sigmoid/', 'Sigmoid/')
        ax.plot(epochs, valacc, '-o', linewidth=1.5, markersize=3, label=label, alpha=0.7)

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Validation Accuracy', fontsize=14)
    ax.set_title('Validation Accuracy Comparison: All Models', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)

    plt.tight_layout()
    output_path = output_dir / "all_models_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Vergleichsplot erstellt: {output_path}")

def create_summary_report(model_data, output_dir):
    """
    Erstellt einen Textbericht mit allen wichtigen Metriken.
    """
    report_path = output_dir / "summary_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("TRAINING SUMMARY REPORT - MNIST Numbers\n")
        f.write("=" * 100 + "\n\n")

        for model_id, data in sorted(model_data.items()):
            if not data:
                continue

            f.write(f"\n{'='*100}\n")
            f.write(f"Modell: {model_id}\n")
            f.write(f"{'='*100}\n")

            # Statistiken
            epochs_trained = len(data)
            final_data = data[-1]
            best_valacc = max([d['valacc'] for d in data])
            best_valacc_epoch = [d['epoch'] for d in data if d['valacc'] == best_valacc][0]
            min_valloss = min([d['valloss'] for d in data])
            min_valloss_epoch = [d['epoch'] for d in data if d['valloss'] == min_valloss][0]

            f.write(f"\nAnzahl trainierter Epochen: {epochs_trained}\n")
            f.write(f"Gesamte Trainingszeit: {final_data['time']:.2f} Sekunden ({final_data['time']/60:.2f} Minuten)\n")
            f.write(f"\nLetzte Epoche ({final_data['epoch']}):\n")
            f.write(f"  - Training Accuracy: {final_data['acc']:.4f}\n")
            f.write(f"  - Validation Accuracy: {final_data['valacc']:.4f}\n")
            f.write(f"  - Validation Loss: {final_data['valloss']:.4f}\n")
            f.write(f"\nBeste Metriken:\n")
            f.write(f"  - Beste Validation Accuracy: {best_valacc:.4f} (Epoche {best_valacc_epoch})\n")
            f.write(f"  - Niedrigster Validation Loss: {min_valloss:.4f} (Epoche {min_valloss_epoch})\n")

            # Overfitting-Check
            final_gap = final_data['acc'] - final_data['valacc']
            if final_gap > 0.05:
                f.write(f"\n⚠ Hinweis: Mögliches Overfitting (Gap: {final_gap:.4f})\n")

            f.write("\n")

    print(f"\n✓ Summary Report erstellt: {report_path}")

def get_model_category(model_id):
    """
    Extrahiert Kategorie aus Model-ID: (activation, batch_mode)
    z.B. 'cpu/relu/batch/BS-32/...' -> ('relu', 'batch')
    """
    parts = model_id.split('\\' if '\\' in model_id else '/')

    activation = 'relu' if 'relu' in parts else 'sigmoid'
    batch_mode = 'batch' if 'batch' in parts else 'nobatch'

    return (activation, batch_mode)

def get_best_models_per_category(model_data, top_n=5):
    """
    Findet die besten Modelle pro Kategorie (ReLU/Sigmoid × Batch/NoBatch).

    Returns:
        Dict mit Kategorien als Keys und Liste der besten Modelle als Values
    """
    categories = defaultdict(list)

    for model_id, data in model_data.items():
        if not data:
            continue

        category = get_model_category(model_id)
        best_valacc = max([d['valacc'] for d in data])

        categories[category].append({
            'model_id': model_id,
            'best_valacc': best_valacc,
            'data': data
        })

    # Sortiere und nimm Top N pro Kategorie
    best_per_category = {}
    for category, models in categories.items():
        sorted_models = sorted(models, key=lambda x: x['best_valacc'], reverse=True)
        best_per_category[category] = sorted_models[:top_n]

    return best_per_category

def create_best_models_report(model_data, output_dir, top_n=5):
    """
    Erstellt einen übersichtlichen Report der besten Modelle pro Kategorie.
    """
    best_per_category = get_best_models_per_category(model_data, top_n)

    report_path = output_dir / "best_models_overview.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write(" " * 40 + "BEST MODELS OVERVIEW - MNIST Numbers\n")
        f.write("=" * 120 + "\n\n")

        category_names = {
            ('relu', 'batch'): '🚀 ReLU + Batch Training',
            ('relu', 'nobatch'): '🐢 ReLU + No Batch',
            ('sigmoid', 'batch'): '📈 Sigmoid + Batch Training',
            ('sigmoid', 'nobatch'): '📉 Sigmoid + No Batch'
        }

        for category in [('relu', 'batch'), ('relu', 'nobatch'), ('sigmoid', 'batch'), ('sigmoid', 'nobatch')]:
            if category not in best_per_category:
                continue

            f.write("\n" + "=" * 120 + "\n")
            f.write(f"{category_names[category]}\n")
            f.write("=" * 120 + "\n\n")

            models = best_per_category[category]

            for i, model_info in enumerate(models, 1):
                model_id = model_info['model_id']
                data = model_info['data']
                best_valacc = model_info['best_valacc']

                final_data = data[-1]
                best_valacc_epoch = [d['epoch'] for d in data if d['valacc'] == best_valacc][0]
                min_valloss = min([d['valloss'] for d in data])

                # Konfiguration extrahieren
                parts = model_id.split('\\' if '\\' in model_id else '/')
                hl = next((p for p in parts if p.startswith('HL-')), 'N/A')
                lr = next((p for p in parts if p.startswith('LR-')), 'N/A')
                bs = next((p for p in parts if p.startswith('BS-')), 'N/A')

                f.write(f"Rang {i}: Val Acc = {best_valacc:.4f} (Epoche {best_valacc_epoch})\n")
                f.write(f"{'─' * 120}\n")
                f.write(f"  Modell: {model_id}\n")
                f.write(f"  Konfiguration: {hl} | {lr} | {bs if bs != 'N/A' else 'No Batch'}\n")
                f.write(f"  \n")
                f.write(f"  Beste Val Accuracy:  {best_valacc:.4f} @ Epoche {best_valacc_epoch}\n")
                f.write(f"  Finale Val Accuracy: {final_data['valacc']:.4f} @ Epoche {final_data['epoch']}\n")
                f.write(f"  Niedrigster Val Loss: {min_valloss:.4f}\n")
                f.write(f"  Trainingszeit: {final_data['time']:.1f}s ({final_data['time']/60:.2f} min)\n")

                # Overfitting Check
                gap = final_data['acc'] - final_data['valacc']
                if gap > 0.05:
                    f.write(f"  ⚠️  Overfitting-Warnung: Gap = {gap:.4f}\n")

                f.write("\n")

    print(f"✓ Best Models Overview erstellt: {report_path}")

def create_best_models_comparison_plot(model_data, output_dir, top_n=5):
    """
    Erstellt Vergleichsplots nur mit den besten Modellen pro Kategorie.
    """
    best_per_category = get_best_models_per_category(model_data, top_n)

    category_names = {
        ('relu', 'batch'): 'ReLU + Batch',
        ('relu', 'nobatch'): 'ReLU + No Batch',
        ('sigmoid', 'batch'): 'Sigmoid + Batch',
        ('sigmoid', 'nobatch'): 'Sigmoid + No Batch'
    }

    colors = {
        ('relu', 'batch'): 'tab:blue',
        ('relu', 'nobatch'): 'tab:cyan',
        ('sigmoid', 'batch'): 'tab:orange',
        ('sigmoid', 'nobatch'): 'tab:red'
    }

    # 1. Großer Vergleichsplot mit allen Top-Modellen
    fig, ax = plt.subplots(figsize=(16, 10))

    for category, models in best_per_category.items():
        color = colors[category]
        cat_name = category_names[category]

        for i, model_info in enumerate(models):
            data = model_info['data']
            epochs = [d['epoch'] for d in data]
            valacc = [d['valacc'] for d in data]

            # Nur ersten mit Label versehen
            label = f"{cat_name} (Top {i+1})" if i < 3 else None
            alpha = 0.9 - (i * 0.15)
            linewidth = 2.5 - (i * 0.3)

            ax.plot(epochs, valacc, color=color, alpha=alpha,
                   linewidth=linewidth, label=label)

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'Best Models Comparison (Top {top_n} per Category)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10, ncol=2)
    ax.set_ylim([0.85, 1.0])

    plt.tight_layout()
    output_path = output_dir / "best_models_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Best Models Comparison erstellt: {output_path}")

    # 2. Subplots für jede Kategorie
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Best Models per Category - Detailed View', fontsize=18, fontweight='bold')

    ax_map = {
        ('relu', 'batch'): axes[0, 0],
        ('relu', 'nobatch'): axes[0, 1],
        ('sigmoid', 'batch'): axes[1, 0],
        ('sigmoid', 'nobatch'): axes[1, 1]
    }

    for category, ax in ax_map.items():
        if category not in best_per_category:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(category_names[category], fontsize=14, fontweight='bold')
            continue

        models = best_per_category[category]

        for i, model_info in enumerate(models):
            data = model_info['data']
            model_id = model_info['model_id']
            epochs = [d['epoch'] for d in data]
            valacc = [d['valacc'] for d in data]

            # Label kürzen
            parts = model_id.split('\\' if '\\' in model_id else '/')
            hl = next((p for p in parts if p.startswith('HL-')), 'N/A')
            lr = next((p for p in parts if p.startswith('LR-')), 'N/A')
            label = f"{hl} {lr} ({model_info['best_valacc']:.4f})"

            ax.plot(epochs, valacc, linewidth=2, label=label, marker='o',
                   markersize=3, alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Validation Accuracy', fontsize=11)
        ax.set_title(category_names[category], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim([0.85, 1.0])

    plt.tight_layout()
    output_path = output_dir / "best_models_by_category.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Best Models by Category erstellt: {output_path}")

def create_training_time_analysis(model_data, output_dir):
    """
    Erstellt detaillierte Analysen zur Trainingszeit:
    - Zeit pro Epoch
    - Gesamttrainingszeit
    - Effizienz (Accuracy pro Zeit)
    - Zeit vs Accuracy Trade-offs

    Args:
        model_data: Dictionary {model_id: [epochs_data]}
        output_dir: Ausgabeverzeichnis
    """
    if not model_data:
        print("⚠️  Keine Daten für Time Analysis")
        return

    print("\n" + "="*100)
    print("TRAININGSZEIT-ANALYSE")
    print("="*100 + "\n")

    # Sammle Zeit-Statistiken
    time_stats = []
    for model_id, data in model_data.items():
        if not data:
            continue

        parts = model_id.split('\\' if '\\' in model_id else '/')

        # Extrahiere Konfiguration
        config = {
            'gpu': 'gpu' in parts,
            'relu': 'relu' in parts,
            'batch': 'batch' in parts,
            'batch_size': next((p.split('-')[1] for p in parts if p.startswith('BS-')), 'N/A'),
            'hidden_layers': next((p for p in parts if p.startswith('HL-')), 'N/A'),
            'lr': next((p for p in parts if p.startswith('LR-')), 'N/A'),
            'seed': next((p for p in parts if p.startswith('S-')), 'N/A'),
        }

        total_epochs = len(data)
        total_time = data[-1]['time'] if data else 0
        avg_time_per_epoch = total_time / total_epochs if total_epochs > 0 else 0

        best_valacc = max([d['valacc'] for d in data])
        best_epoch = next((i+1 for i, d in enumerate(data) if d['valacc'] == best_valacc), 0)
        time_to_best = next((d['time'] for d in data if d['valacc'] == best_valacc), 0)

        # Effizienz-Metriken
        efficiency = best_valacc / total_time if total_time > 0 else 0  # Acc/s
        time_per_percent_acc = total_time / (best_valacc * 100) if best_valacc > 0 else 0

        time_stats.append({
            'model_id': model_id,
            'config': config,
            'total_epochs': total_epochs,
            'total_time': total_time,
            'avg_time_per_epoch': avg_time_per_epoch,
            'best_valacc': best_valacc,
            'best_epoch': best_epoch,
            'time_to_best': time_to_best,
            'efficiency': efficiency,
            'time_per_percent_acc': time_per_percent_acc,
            'data': data
        })

    # Sortiere nach verschiedenen Kriterien
    fastest_models = sorted(time_stats, key=lambda x: x['total_time'])[:10]
    most_efficient = sorted(time_stats, key=lambda x: x['efficiency'], reverse=True)[:10]
    fastest_to_best = sorted(time_stats, key=lambda x: x['time_to_best'])[:10]

    # 1. TEXT REPORT
    report_path = output_dir / "training_time_analysis.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("TRAININGSZEIT-ANALYSE\n")
        f.write("="*100 + "\n\n")

        # Schnellste Modelle (Gesamtzeit)
        f.write("🚀 TOP 10 SCHNELLSTE MODELLE (Gesamttrainingszeit)\n")
        f.write("-"*100 + "\n")
        for i, stats in enumerate(fastest_models, 1):
            f.write(f"{i:2d}. {stats['model_id']}\n")
            f.write(f"    Gesamtzeit: {stats['total_time']:.1f}s ({stats['total_time']/60:.1f}min)\n")
            f.write(f"    Epochs: {stats['total_epochs']}, Ø Zeit/Epoch: {stats['avg_time_per_epoch']:.2f}s\n")
            f.write(f"    Best Val Acc: {stats['best_valacc']:.4f} (Epoch {stats['best_epoch']})\n")
            f.write("\n")

        # Effizienteste Modelle (Accuracy pro Zeit)
        f.write("\n" + "="*100 + "\n")
        f.write("⚡ TOP 10 EFFIZIENTESTE MODELLE (Val Accuracy pro Sekunde)\n")
        f.write("-"*100 + "\n")
        for i, stats in enumerate(most_efficient, 1):
            f.write(f"{i:2d}. {stats['model_id']}\n")
            f.write(f"    Effizienz: {stats['efficiency']:.6f} (Val Acc / Sekunde)\n")
            f.write(f"    Best Val Acc: {stats['best_valacc']:.4f} in {stats['total_time']:.1f}s\n")
            f.write(f"    Zeit pro 1% Accuracy: {stats['time_per_percent_acc']:.2f}s\n")
            f.write("\n")

        # Schnellste bis zum besten Modell
        f.write("\n" + "="*100 + "\n")
        f.write("🎯 TOP 10 SCHNELLSTE ZUM BESTEN ERGEBNIS\n")
        f.write("-"*100 + "\n")
        for i, stats in enumerate(fastest_to_best, 1):
            f.write(f"{i:2d}. {stats['model_id']}\n")
            f.write(f"    Zeit zum Best: {stats['time_to_best']:.1f}s ({stats['time_to_best']/60:.1f}min)\n")
            f.write(f"    Best Val Acc: {stats['best_valacc']:.4f} (Epoch {stats['best_epoch']}/{stats['total_epochs']})\n")
            f.write(f"    Restliche Zeit: {stats['total_time'] - stats['time_to_best']:.1f}s "
                   f"({(stats['total_time'] - stats['time_to_best'])/stats['total_time']*100:.1f}% verschwendet)\n")
            f.write("\n")

        # Aggregierte Statistiken nach Kategorien
        f.write("\n" + "="*100 + "\n")
        f.write("📊 DURCHSCHNITTLICHE TRAININGSZEITEN NACH KONFIGURATION\n")
        f.write("-"*100 + "\n\n")

        # Gruppiere nach Batch/NoBatch
        batch_times = [s for s in time_stats if s['config']['batch']]
        nobatch_times = [s for s in time_stats if not s['config']['batch']]

        if batch_times:
            avg_batch = sum(s['avg_time_per_epoch'] for s in batch_times) / len(batch_times)
            f.write(f"BATCH Training:\n")
            f.write(f"  Ø Zeit/Epoch: {avg_batch:.2f}s ({len(batch_times)} Modelle)\n\n")

        if nobatch_times:
            avg_nobatch = sum(s['avg_time_per_epoch'] for s in nobatch_times) / len(nobatch_times)
            f.write(f"NO-BATCH Training:\n")
            f.write(f"  Ø Zeit/Epoch: {avg_nobatch:.2f}s ({len(nobatch_times)} Modelle)\n\n")

        # Gruppiere nach ReLU/Sigmoid
        relu_times = [s for s in time_stats if s['config']['relu']]
        sigmoid_times = [s for s in time_stats if not s['config']['relu']]

        if relu_times:
            avg_relu = sum(s['avg_time_per_epoch'] for s in relu_times) / len(relu_times)
            f.write(f"ReLU Aktivierung:\n")
            f.write(f"  Ø Zeit/Epoch: {avg_relu:.2f}s ({len(relu_times)} Modelle)\n\n")

        if sigmoid_times:
            avg_sigmoid = sum(s['avg_time_per_epoch'] for s in sigmoid_times) / len(sigmoid_times)
            f.write(f"Sigmoid Aktivierung:\n")
            f.write(f"  Ø Zeit/Epoch: {avg_sigmoid:.2f}s ({len(sigmoid_times)} Modelle)\n\n")

    print(f"✓ Trainingszeit-Report erstellt: {report_path}")

    # 2. VISUALISIERUNG: Zeit vs Accuracy Trade-off
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Gesamtzeit vs Best Val Accuracy
    total_times = [s['total_time'] for s in time_stats]
    best_accs = [s['best_valacc'] for s in time_stats]
    colors = ['green' if s['config']['batch'] else 'blue' for s in time_stats]

    ax1.scatter(total_times, best_accs, c=colors, alpha=0.6, s=50)
    ax1.set_xlabel('Gesamttrainingszeit (Sekunden)', fontsize=11)
    ax1.set_ylabel('Best Validation Accuracy', fontsize=11)
    ax1.set_title('Trainingszeit vs Accuracy', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Batch', 'No-Batch'], loc='lower right')

    # Plot 2: Zeit pro Epoch (Box-Plot nach Konfiguration)
    batch_epoch_times = [s['avg_time_per_epoch'] for s in batch_times] if batch_times else []
    nobatch_epoch_times = [s['avg_time_per_epoch'] for s in nobatch_times] if nobatch_times else []

    data_to_plot = []
    labels = []
    if batch_epoch_times:
        data_to_plot.append(batch_epoch_times)
        labels.append('Batch')
    if nobatch_epoch_times:
        data_to_plot.append(nobatch_epoch_times)
        labels.append('No-Batch')

    if data_to_plot:
        ax2.boxplot(data_to_plot, labels=labels)
        ax2.set_ylabel('Zeit pro Epoch (Sekunden)', fontsize=11)
        ax2.set_title('Zeit pro Epoch nach Trainingsmodus', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Effizienz (Accuracy/Zeit)
    efficiencies = [s['efficiency'] for s in time_stats]
    model_labels = [s['model_id'].split('/')[-1] if '/' in s['model_id']
                   else s['model_id'].split('\\')[-1] for s in time_stats]

    top_10_eff = sorted(time_stats, key=lambda x: x['efficiency'], reverse=True)[:10]
    eff_vals = [s['efficiency'] for s in top_10_eff]
    eff_labels = [f"{s['config']['hidden_layers']}-{s['config']['lr']}" for s in top_10_eff]

    ax3.barh(range(len(eff_vals)), eff_vals, color='orange', alpha=0.7)
    ax3.set_yticks(range(len(eff_labels)))
    ax3.set_yticklabels(eff_labels, fontsize=8)
    ax3.set_xlabel('Effizienz (Val Acc / Sekunde)', fontsize=11)
    ax3.set_title('Top 10 Effizienteste Modelle', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()

    # Plot 4: Zeit bis zum besten Modell vs Gesamtzeit
    time_to_best_list = [s['time_to_best'] for s in time_stats]
    wasted_time = [s['total_time'] - s['time_to_best'] for s in time_stats]
    wasted_percent = [(s['total_time'] - s['time_to_best'])/s['total_time']*100
                     for s in time_stats if s['total_time'] > 0]

    ax4.scatter(total_times, time_to_best_list, c=wasted_percent,
               cmap='RdYlGn_r', alpha=0.6, s=50)
    ax4.plot([0, max(total_times)], [0, max(total_times)],
            'k--', alpha=0.3, label='Ideal (Best = Ende)')
    ax4.set_xlabel('Gesamttrainingszeit (Sekunden)', fontsize=11)
    ax4.set_ylabel('Zeit bis zum besten Modell (Sekunden)', fontsize=11)
    ax4.set_title('Früherkennung: Wann wurde das beste Modell erreicht?', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Verschwendete Zeit (%)', fontsize=10)

    fig.suptitle('Trainingszeit-Analyse: Performance vs Effizienz', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "training_time_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Trainingszeit-Grafik erstellt: {output_path}")

    # 3. ZUSÄTZLICHE VISUALISIERUNG: Zeitverlauf über Epochen
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Zeige Top 10 beste Modelle und ihre Zeit-Entwicklung
    top_models = sorted(time_stats, key=lambda x: x['best_valacc'], reverse=True)[:10]

    for stats in top_models:
        data = stats['data']
        epochs = [d['epoch'] for d in data]
        times = [d['time'] for d in data]

        # Berechne Zeit-Deltas (Zeit pro Epoch)
        time_per_epoch = [times[0]] + [times[i] - times[i-1] for i in range(1, len(times))]

        label = f"{stats['config']['hidden_layers']} ({stats['best_valacc']:.4f})"
        ax.plot(epochs, time_per_epoch, linewidth=2, label=label, marker='o',
               markersize=3, alpha=0.7)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Zeit pro Epoch (Sekunden)', fontsize=11)
    ax.set_title('Zeitverlauf: Top 10 Modelle - Zeit pro Epoch', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    output_path = output_dir / "training_time_per_epoch.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Zeit-pro-Epoch-Grafik erstellt: {output_path}")

    print("\n" + "="*100)
    print("✓ TRAININGSZEIT-ANALYSE ABGESCHLOSSEN")
    print("="*100)

def main(min_valacc_filter=0.90, top_n_per_category=5, create_all_plots=False):
    """
    Hauptfunktion: Sammelt Daten und erstellt alle Visualisierungen.

    Args:
        min_valacc_filter: Minimale Val Accuracy für individuelle Plots (0.0 = keine Filterung)
        top_n_per_category: Anzahl der besten Modelle pro Kategorie im Best Models Report
        create_all_plots: Wenn False, werden nur Plots für gute Modelle erstellt
    """
    print("\n" + "="*100)
    print("MNIST Training Progress Analyzer (Enhanced)")
    print("="*100 + "\n")

    # Output-Verzeichnis erstellen
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output-Verzeichnis: {OUTPUT_DIR}\n")

    # Daten sammeln (ALLE Modelle für Statistiken)
    print("Sammle alle Trainingsdaten...")
    all_model_data = collect_model_data(min_valacc=0.0)
    print(f"✓ {len(all_model_data)} Modellkonfigurationen gefunden\n")

    if not all_model_data:
        print("❌ Keine Trainingsdaten gefunden!")
        return

    # BESTE MODELLE ANALYSE (WICHTIGSTER TEIL)
    print("="*100)
    print("BEST MODELS ANALYSE")
    print("="*100 + "\n")

    # Best Models Report erstellen
    print(f"Erstelle Best Models Report (Top {top_n_per_category} pro Kategorie)...")
    create_best_models_report(all_model_data, OUTPUT_DIR, top_n=top_n_per_category)

    # Best Models Vergleichsplot erstellen
    print(f"Erstelle Best Models Vergleichsplots...")
    create_best_models_comparison_plot(all_model_data, OUTPUT_DIR, top_n=top_n_per_category)

    # TRAININGSZEIT-ANALYSE (NEU!)
    print(f"\nErstelle Trainingszeit-Analyse...")
    create_training_time_analysis(all_model_data, OUTPUT_DIR)

    # GEFILTERTE MODELLE FÜR DETAILANALYSE
    if min_valacc_filter > 0:
        print(f"\n{'='*100}")
        print(f"GEFILTERTE MODELLE (Val Acc >= {min_valacc_filter:.2f})")
        print(f"{'='*100}\n")

        filtered_model_data = collect_model_data(min_valacc=min_valacc_filter)
        print(f"✓ {len(filtered_model_data)} gute Modelle gefunden (von {len(all_model_data)})\n")

        # Einzelne Plots nur für gute Modelle
        if filtered_model_data:
            print("Erstelle individuelle Plots für gute Modelle...")
            for model_id, data in sorted(filtered_model_data.items()):
                create_model_plot(model_id, data, OUTPUT_DIR)

        # Vergleichsplot für gefilterte Modelle
        print("\nErstelle Vergleichsplot für gute Modelle...")
        create_comparison_plot(filtered_model_data, OUTPUT_DIR)

        # Summary Report für gefilterte Modelle
        print("\nErstelle Summary Report für gute Modelle...")
        create_summary_report(filtered_model_data, OUTPUT_DIR)

    elif create_all_plots:
        # ALLE MODELLE (falls gewünscht)
        print(f"\n{'='*100}")
        print("ALLE MODELLE (keine Filterung)")
        print(f"{'='*100}\n")

        print("Erstelle individuelle Plots für ALLE Modelle...")
        for model_id, data in sorted(all_model_data.items()):
            create_model_plot(model_id, data, OUTPUT_DIR)

        print("\nErstelle Vergleichsplot für alle Modelle...")
        create_comparison_plot(all_model_data, OUTPUT_DIR)

        print("\nErstelle Summary Report für alle Modelle...")
        create_summary_report(all_model_data, OUTPUT_DIR)

    print("\n" + "="*100)
    print("✓ FERTIG! Alle Visualisierungen wurden erstellt.")
    print("="*100)
    print(f"\nWICHTIG: Schaue dir zuerst an:")
    print(f"  📊 {OUTPUT_DIR / 'best_models_overview.txt'}")
    print(f"  📈 {OUTPUT_DIR / 'best_models_comparison.png'}")
    print(f"  📉 {OUTPUT_DIR / 'best_models_by_category.png'}")
    print(f"\nNEU - Trainingszeit-Analyse:")
    print(f"  ⏱️  {OUTPUT_DIR / 'training_time_analysis.txt'}")
    print(f"  📊 {OUTPUT_DIR / 'training_time_analysis.png'}")
    print(f"  📈 {OUTPUT_DIR / 'training_time_per_epoch.png'}")
    print("="*100 + "\n")

if __name__ == "__main__":
    # KONFIGURATION:
    # - min_valacc_filter: Nur Modelle mit Val Acc >= X werden im Detail analysiert
    # - top_n_per_category: Top N Modelle pro Kategorie (ReLU/Sigmoid × Batch/NoBatch)
    # - create_all_plots: True = alle Plots, False = nur gefilterte Plots

    main(
        min_valacc_filter=0.92,  # Nur Modelle mit >= 92% Val Accuracy
        top_n_per_category=5,     # Top 5 pro Kategorie
        create_all_plots=False    # Keine Plots für schlechte Modelle
    )
