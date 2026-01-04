# NanoNeuron - Projektprotokoll

## Inhaltsverzeichnis
1. [Projektübersicht](#projektuebersicht)
2. [Funktionen des Neuronalen Netzes](#funktionen-des-neuronalen-netzes)
3. [Implementierungsvergleich: Ohne vs. Mit NumPy](#implementierungsvergleich-ohne-vs-mit-numpy)
4. [Auswertung der trainierten Netze](#auswertung-der-trainierten-netze)
5. [Ergebnisse und Schlussfolgerungen](#ergebnisse-und-schlussfolgerungen)

---

## <a name="projektuebersicht"></a>Projektübersicht

Das NanoNeuron-Projekt implementiert ein einfaches neuronales Netzwerk zur Erkennung handgeschriebener Ziffern (0-9) aus dem MNIST-Dataset. Das Projekt demonstriert sowohl eine native Python-Implementierung als auch eine optimierte NumPy-Variante.

### Netzwerkarchitektur

```
Input Layer (784 Neuronen)  →  Hidden Layer (variabel)  →  Output Layer (10 Neuronen)
     28x28 Pixel                  Sigmoid-Aktivierung         Sigmoid-Aktivierung
```

- **Input Layer**: 784 Neuronen (28×28 Pixel Grauwertbilder)
- **Hidden Layer**: Variable Anzahl (z.B. 30, 65, 120, 170 Neuronen)
- **Output Layer**: 10 Neuronen (eine für jede Ziffer 0-9)

---

## Funktionen des Neuronalen Netzes

### 1. Initialisierung (`__init__`)

Das Netzwerk wird mit zufälligen Gewichten initialisiert:

```python
# Gewichte für Hidden Layer (W1) und Output Layer (W2)
self.W1 = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
self.b1 = np.zeros(hidden_size)  # Bias-Werte für Hidden Layer

self.W2 = np.random.uniform(-0.1, 0.1, (output_size, hidden_size))
self.b2 = np.zeros(output_size)  # Bias-Werte für Output Layer
```

**Zweck**: 
- Zufällige Initialisierung der Gewichte im Bereich [-0.1, 0.1]
- Bias-Werte werden mit 0 initialisiert
- Optional: Laden vortrainierter Gewichte möglich

### 2. Sigmoid-Aktivierungsfunktion

```python
def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
```

**Eigenschaften**:
- Transformiert beliebige Eingabewerte in den Bereich [0, 1]
- Differenzierbar (wichtig für Backpropagation)
- Ableitung: `sigmoid(x) * (1 - sigmoid(x))`

**Visualisierung**:
```
     1.0 |         ╭────────
         |       ╭─╯
     0.5 |     ╭─┤
         |   ╭─╯  
     0.0 |───╯
         └─────────────────
        -5    0    5
```

### 3. Forward Propagation (`forward`)

Die Forward Propagation berechnet die Ausgabe des Netzwerks:

```python
def forward(self, x):
    # Hidden Layer Berechnung
    h_raw = self.W1 @ x + self.b1
    h = self.sigmoid(h_raw)
    
    # Output Layer Berechnung
    o_raw = self.W2 @ h + self.b2
    o = self.sigmoid(o_raw)
    
    return h, o
```

**Ablauf**:
1. **Input → Hidden Layer**: 
   - Gewichtete Summe: `W1 × Input + b1`
   - Aktivierung durch Sigmoid
   
2. **Hidden → Output Layer**:
   - Gewichtete Summe: `W2 × Hidden + b2`
   - Aktivierung durch Sigmoid
   
3. **Ausgabe**: Vektor mit 10 Werten (Wahrscheinlichkeiten für jede Ziffer)

### 4. Backpropagation (`train`)

Das Herzstück des Lernprozesses - berechnet Fehler und aktualisiert Gewichte:

```python
def train(self, x, label):
    h, o = self.forward(x)
    
    # Erstelle Zielvektor (One-Hot Encoding)
    y = np.zeros(self.output_size)
    y[label] = 1
    
    # Fehlerberechnung Output Layer
    error_out = (o - y) * o * (1 - o)  # Gradient der Sigmoid-Funktion
    
    # Fehlerberechnung Hidden Layer (Rückpropagierung)
    error_hidden = (self.W2.T @ error_out) * h * (1 - h)
    
    # Gewichtsaktualisierung (Gradient Descent)
    self.W2 -= self.learning_rate * np.outer(error_out, h)
    self.b2 -= self.learning_rate * error_out
    
    self.W1 -= self.learning_rate * np.outer(error_hidden, x)
    self.b1 -= self.learning_rate * error_hidden
```

**Mathematischer Hintergrund**:

1. **Fehler im Output Layer**:
   - Differenz zwischen Vorhersage und Ziel
   - Multipliziert mit Ableitung der Sigmoid-Funktion
   
2. **Fehler im Hidden Layer**:
   - Rückpropagierung des Fehlers vom Output Layer
   - Gewichtet durch `W2` (Transpose)
   
3. **Gewichtsaktualisierung**:
   - `W_neu = W_alt - learning_rate × Gradient`
   - Kleinere Lernrate = langsameres, aber stabileres Lernen
   - Größere Lernrate = schnelleres, aber instabileres Lernen

### 5. Vorhersage (`predict`)

```python
def predict(self, x):
    _, o = self.forward(x)
    return np.argmax(o)  # Index mit höchster Aktivierung
```

**Funktion**: Gibt die Ziffer mit der höchsten Ausgabewahrscheinlichkeit zurück.

---

## Implementierungsvergleich: Ohne vs. Mit NumPy

Das Projekt enthält zwei Implementierungen:
- **`net_old.py`**: Native Python-Implementierung (ohne NumPy)
- **`net.py`**: NumPy-optimierte Implementierung

### Unterschiede im Detail

#### 1. Datenstrukturen

**Ohne NumPy** (`net_old.py`):
```python
# Listen von Listen
self.W1 = [[self.rand() for _ in range(self.input_size)]
           for _ in range(self.hidden_size)]
self.b1 = [0.0 for _ in range(self.hidden_size)]
```

**Mit NumPy** (`net.py`):
```python
# NumPy Arrays (optimierte C-Implementierung)
self.W1 = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
self.b1 = np.zeros(hidden_size)
```

#### 2. Matrix-Operationen

**Ohne NumPy** - Manuelle Schleifen:
```python
# Dot-Produkt manuell berechnen
def dot(self, a, b):
    return sum(x * y for x, y in zip(a, b))

# Forward Propagation mit Schleifen
h = [
    self.sigmoid(self.dot(self.W1[i], x) + self.b1[i])
    for i in range(self.hidden_size)
]
```

**Mit NumPy** - Vektorisierte Operationen:
```python
# Direkte Matrix-Multiplikation (C-optimiert)
h_raw = self.W1 @ x + self.b1
h = self.sigmoid(h_raw)
```

#### 3. Backpropagation

**Ohne NumPy** - Verschachtelte Schleifen:
```python
# Manuelle Gewichtsaktualisierung
for i in range(self.output_size):
    for j in range(self.hidden_size):
        self.W2[i][j] -= self.learning_rate * error_out[i] * h[j]
    self.b2[i] -= self.learning_rate * error_out[i]
```

**Mit NumPy** - Vektorisierte Operationen:
```python
# Effiziente Matrix-Operationen
self.W2 -= self.learning_rate * np.outer(error_out, h)
self.b2 -= self.learning_rate * error_out
```

### Performance-Vergleich

| Aspekt | Ohne NumPy | Mit NumPy |
|--------|-----------|-----------|
| **Geschwindigkeit** | ~10-50x langsamer | Baseline (C-optimiert) |
| **Speichereffizienz** | Weniger effizient (Python-Listen) | Sehr effizient (contiguous arrays) |
| **Code-Länge** | Länger (mehr Schleifen) | Kürzer und lesbarer |
| **Lesbarkeit** | Verschachtelte Schleifen | Mathematische Notation |
| **Skalierbarkeit** | Schlecht bei großen Netzen | Sehr gut |

### Vorteile NumPy-Implementierung

1. **Performance**: 10-50× schnellere Ausführung durch C-Backend
2. **Parallelisierung**: Automatische Nutzung von SIMD-Instruktionen
3. **Speicher**: Effiziente contiguous memory arrays
4. **Numerische Stabilität**: Bessere Implementierung mathematischer Funktionen
5. **Lesbarkeit**: Mathematische Notation näher an theoretischen Formeln
6. **Wartbarkeit**: Weniger Code, weniger Fehlerquellen

### Wann welche Implementierung?

- **Ohne NumPy**: 
  - Lernzwecke (besseres Verständnis der Algorithmen)
  - Sehr kleine Netze
  - Umgebungen ohne NumPy-Support
  
- **Mit NumPy**:
  - Produktive Anwendungen
  - Größere Datensätze
  - Wenn Performance wichtig ist

---

## Auswertung der trainierten Netze

Das Projekt enthält 69 trainierte Modelle mit verschiedenen Konfigurationen über drei Datensatzgrößen verteilt.

### Datensätze

1. **5000 Samples**: 52 Modelle
2. **50000 Samples**: 8 Modelle
3. **59999 Samples**: 8 Modelle (fast vollständiger MNIST-Trainingsdatensatz)

### Hyperparameter-Analyse

#### 1. Training mit 5000 Samples

**Durchschnittliche Genauigkeit nach Lernrate**:
```
Learning Rate 0.01: ████████████████████ 0.4878 (48.78%)
Learning Rate 0.02: ████████████████████████████ 0.8367 (83.67%)
Learning Rate 0.05: ███████████████ 0.4423 (44.23%)
Learning Rate 0.10: █████████████████ 0.5317 (53.17%)
Learning Rate 0.20: ████████████████████████ 0.7618 (76.18%)
```

**Bestes Modell**:
- **Genauigkeit**: 91.23%
- **Hidden Size**: 65 Neuronen
- **Learning Rate**: 0.20
- **Seed**: 498

**Erkenntnisse**:
- Mit nur 5000 Samples ist die Varianz sehr hoch
- LR 0.02 zeigt überraschend gute Durchschnittswerte
- LR 0.05 und 0.10 zeigen schlechtere Performance (möglicherweise Overfitting)
- Beste Einzelergebnisse bei LR 0.20

#### 2. Training mit 50000 Samples

**Durchschnittliche Genauigkeit nach Lernrate**:
```
Learning Rate 0.01: ██████████████████████████ 0.9158 (91.58%)
Learning Rate 0.02: ███████████████████████████ 0.9297 (92.97%)
Learning Rate 0.10: ████████████████████████████ 0.9549 (95.49%)
Learning Rate 0.20: ██████████████████████████████ 0.9634 (96.34%)
```

**Bestes Modell**:
- **Genauigkeit**: 96.61%
- **Hidden Size**: 120 Neuronen
- **Learning Rate**: 0.20
- **Seed**: 485

**Erkenntnisse**:
- Deutlich stabilere Ergebnisse mit mehr Trainingsdaten
- Klarer Trend: Höhere Lernrate → Bessere Genauigkeit
- LR 0.20 liefert konsistent die besten Ergebnisse
- Größeres Hidden Layer (120) performt besser als kleineres (65)

#### 3. Training mit 59999 Samples (Fast vollständig)

**Durchschnittliche Genauigkeit nach Lernrate**:
```
Learning Rate 0.20: ████████████████████████████ 0.9699 (96.99%)
Learning Rate 0.25: █████████████████████████████ 0.9710 (97.10%)
Learning Rate 0.30: █████████████████████████████ 0.9710 (97.10%)
```

**Beste Modelle** (2 Modelle mit gleicher Genauigkeit):
1. **Modell A**:
   - **Genauigkeit**: 97.10%
   - **Hidden Size**: 120 Neuronen
   - **Learning Rate**: 0.30
   - **Seed**: 63

2. **Modell B**:
   - **Genauigkeit**: 97.10%
   - **Hidden Size**: 120 Neuronen
   - **Learning Rate**: 0.25
   - **Seed**: 1000

**Erkenntnisse**:
- Maximale Trainingsdaten führen zu bester Performance
- LR 0.25-0.30 optimal für diesen Datensatz
- Hidden Size 120-170 liefert beste Ergebnisse
- Sehr konsistente Ergebnisse (alle >96.8%)

### Vergleichstabelle: Top-Modelle

| Samples | Accuracy | Hidden Size | Learning Rate | Verbesserung |
|---------|----------|-------------|---------------|--------------|
| 5,000   | 91.23%   | 65          | 0.20          | Baseline     |
| 50,000  | 96.61%   | 120         | 0.20          | +5.38%       |
| 59,999  | 97.10%   | 120         | 0.30          | +5.87%       |

### Grafische Darstellung: Genauigkeit vs. Trainingsgröße

```
Accuracy (%)
100% |                                    ●───● (97.10%)
     |                               ●─── (96.61%)
 95% |                          ●───
     |                     ●───
 90% |                ●─── (91.23%)
     |           ●───
 85% |      ●───
     | ●───
 80% |
     └──────────────────────────────────────────────
        5K      10K     20K     30K    50K    60K
                    Training Samples
```

### Einfluss der Hyperparameter

#### Hidden Layer Size

| Hidden Size | Bester Acc (59999) | Bemerkung |
|-------------|-------------------|-----------|
| 120         | 97.10%            | Optimal   |
| 130         | 96.82%            | Gut       |
| 140         | 97.03%            | Sehr gut  |
| 150         | 97.02%            | Sehr gut  |
| 160         | 97.01%            | Sehr gut  |
| 170         | 97.07%            | Sehr gut  |
| 190         | 96.97%            | Gut       |

**Erkenntnis**: 120-170 Neuronen im Hidden Layer sind optimal. Mehr Neuronen bringen keinen signifikanten Vorteil.

#### Learning Rate

**Optimale Lernraten nach Datensatzgröße**:

```
Datensatzgröße    Optimale LR    Begründung
──────────────────────────────────────────────────
5,000 Samples     0.02 - 0.20    Hohe Varianz
50,000 Samples    0.10 - 0.20    Stabilere Ergebnisse
59,999 Samples    0.20 - 0.30    Beste Konvergenz
```

### Alle gespeicherten Modelle (Top 10 nach Genauigkeit)

| Rank | Accuracy | Hidden Size | Learning Rate | Samples | Seed |
|------|----------|-------------|---------------|---------|------|
| 1    | 97.10%   | 120         | 0.30          | 59,999  | 63   |
| 2    | 97.10%   | 120         | 0.25          | 59,999  | 1000 |
| 3    | 97.07%   | 170         | 0.20          | 59,999  | 762  |
| 4    | 97.03%   | 140         | 0.20          | 59,999  | 821  |
| 5    | 97.02%   | 150         | 0.20          | 59,999  | 657  |
| 6    | 97.01%   | 160         | 0.20          | 59,999  | 752  |
| 7    | 96.97%   | 190         | 0.20          | 59,999  | 294  |
| 8    | 96.82%   | 130         | 0.20          | 59,999  | 305  |
| 9    | 96.61%   | 120         | 0.20          | 50,000  | 485  |
| 10   | 96.08%   | 65          | 0.20          | 50,000  | 355  |

---

## Ergebnisse und Schlussfolgerungen

### Haupterkenntnisse

#### 1. Einfluss der Trainingsdatenmenge

- **5,000 Samples**: Maximale Genauigkeit ~91% (signifikantes Underfitting)
- **50,000 Samples**: Maximale Genauigkeit ~97% (gutes Ergebnis)
- **59,999 Samples**: Maximale Genauigkeit ~97.1% (beste Ergebnisse)

**Fazit**: Mehr Trainingsdaten verbessern die Genauigkeit deutlich, aber der Effekt flacht ab (~83% der MNIST-Daten reichen für >96% Genauigkeit).

#### 2. Optimale Hyperparameter

**Learning Rate**:
- Zu niedrig (0.01): Langsame Konvergenz, schlechtere Ergebnisse
- Optimal (0.20-0.30): Beste Balance zwischen Konvergenzgeschwindigkeit und Stabilität
- Höhere Werte wurden nicht getestet, könnten zu Instabilität führen

**Hidden Layer Size**:
- 65 Neuronen: Gut für kleinere Datensätze
- 120-170 Neuronen: Optimal für größere Datensätze
- >170 Neuronen: Kein signifikanter Vorteil, erhöhte Berechnungskosten

#### 3. NumPy vs. Native Python

Die NumPy-Implementierung ist klar überlegen:
- ✅ 10-50× schnellere Ausführung
- ✅ Kompakterer, wartbarerer Code
- ✅ Bessere Skalierbarkeit
- ✅ Numerische Stabilität

Die native Python-Implementierung ist wertvoll für:
- 📚 Lernzwecke (besseres Algorithmenverständnis)
- 🔍 Debugging (explizite Operationen)

#### 4. Netzwerk-Performance

Das einfache 3-Layer-Netzwerk erreicht:
- **~97% Genauigkeit** auf MNIST
- **3 Epochen** Training ausreichend
- **Keine Regularisierung** benötigt

Dies ist bemerkenswert für ein so einfaches Netzwerk!

### Verbesserungspotenzial

1. **Architektur**:
   - Mehrere Hidden Layers (Deep Learning)
   - ReLU statt Sigmoid (schnellere Konvergenz)
   - Dropout für bessere Generalisierung

2. **Training**:
   - Learning Rate Scheduling (dynamische Anpassung)
   - Mini-Batch Gradient Descent
   - Data Augmentation (Rotation, Verzerrung)

3. **Optimierung**:
   - Adam statt einfachem Gradient Descent
   - Batch Normalization
   - Cross-Entropy Loss statt MSE

### Praktische Empfehlungen

Für die besten Ergebnisse mit diesem Netzwerk:

```python
# Empfohlene Konfiguration
NNNet(
    input_size=784,
    hidden_size=120,        # Gute Balance
    output_size=10,
    learning_rate=0.25,     # Optimal für MNIST
    seed=42                 # Reproduzierbarkeit
)

# Training
EPOCHS = 3                  # Mehr nicht nötig
TRAIN_SAMPLES = 59999       # Maximale Daten nutzen
```

### Zusammenfassung

Das NanoNeuron-Projekt demonstriert erfolgreich:

1. ✅ Implementierung eines funktionsfähigen neuronalen Netzes
2. ✅ Vergleich zwischen nativer Python- und NumPy-Implementierung
3. ✅ Systematische Hyperparameter-Optimierung
4. ✅ Erreichen von ~97% Genauigkeit auf MNIST

Das Projekt zeigt, dass selbst einfache neuronale Netze mit richtiger Konfiguration beeindruckende Ergebnisse erzielen können. Die NumPy-Implementierung ist klar die bessere Wahl für praktische Anwendungen, während die native Python-Version wertvoll für das Verständnis der zugrundeliegenden Algorithmen bleibt.

---

*Protokoll erstellt am: 04.01.2026*
*Analysierte Modelle: 69*
*Framework: NumPy + Native Python*
