# NanoNeuron Projekt - Protokoll

## Projektübersicht

Das NanoNeuron-Projekt implementiert ein einfaches neuronales Netzwerk zur Erkennung handgeschriebener Ziffern (MNIST-Dataset). Das Projekt demonstriert zwei verschiedene Implementierungsansätze:

1. **net_old.py**: Implementierung ohne NumPy (reine Python-Listen und Schleifen)
2. **net.py**: Optimierte Implementierung mit NumPy-Arrays

## Architektur des Neuronalen Netzwerks

### Netzwerk-Struktur

Das neuronale Netzwerk besteht aus drei Schichten:

- **Input Layer (Eingabeschicht)**: 784 Neuronen (28x28 Pixel eines Bildes)
- **Hidden Layer (Versteckte Schicht)**: Konfigurierbar (Standard: 30-170 Neuronen)
- **Output Layer (Ausgabeschicht)**: 10 Neuronen (für Ziffern 0-9)

```
Input (784) → Hidden Layer (30-170) → Output (10)
```

### Gewichte und Biases

Das Netzwerk verwendet zwei Gewichtsmatrizen und zwei Bias-Vektoren:

- **W1**: Gewichte zwischen Input- und Hidden-Layer (Dimension: hidden_size × input_size)
- **b1**: Biases für die Hidden-Layer (Dimension: hidden_size)
- **W2**: Gewichte zwischen Hidden- und Output-Layer (Dimension: output_size × hidden_size)
- **b2**: Biases für die Output-Layer (Dimension: output_size)

Alle Gewichte werden zufällig zwischen -0.1 und 0.1 initialisiert, während die Biases mit 0 initialisiert werden.

## Hauptfunktionen des Neuronalen Netzwerks

### 1. Sigmoid-Aktivierungsfunktion

```python
def sigmoid(self, x):
    return 1 / (1 + exp(-x))
```

**Zweck**: Die Sigmoid-Funktion transformiert beliebige Eingabewerte in einen Bereich zwischen 0 und 1. Dies ist wichtig für:
- Normalisierung der Neuron-Aktivierungen
- Interpretation der Ausgaben als Wahrscheinlichkeiten
- Ermöglichung des Lernens durch ihre differenzierbare Form

**Eigenschaften**:
- Ausgabebereich: (0, 1)
- Symmetrisch um den Punkt (0, 0.5)
- Ableitung: σ'(x) = σ(x) · (1 - σ(x))

### 2. Forward Propagation (Vorwärtsdurchlauf)

```python
def forward(self, x):
    # Hidden Layer Berechnung
    h_raw = W1 @ x + b1        # Gewichtete Summe
    h = sigmoid(h_raw)          # Aktivierung
    
    # Output Layer Berechnung
    o_raw = W2 @ h + b2        # Gewichtete Summe
    o = sigmoid(o_raw)          # Aktivierung
    
    return h, o
```

**Ablauf**:
1. **Input zu Hidden Layer**:
   - Berechnung der gewichteten Summe: z = W1 · x + b1
   - Anwendung der Sigmoid-Funktion: h = σ(z)

2. **Hidden zu Output Layer**:
   - Berechnung der gewichteten Summe: z = W2 · h + b2
   - Anwendung der Sigmoid-Funktion: o = σ(z)

3. **Ergebnis**: Ein Vektor mit 10 Werten (je einer für jede Ziffer), wobei höhere Werte eine höhere Wahrscheinlichkeit darstellen.

### 3. Backpropagation (Rückwärtsdurchlauf) und Training

```python
def train(self, x, label):
    h, o = self.forward(x)
    
    # One-Hot-Encoding des Labels
    y = zeros(output_size)
    y[label] = 1
    
    # Fehlerberechnung Output Layer
    error_out = (o - y) * o * (1 - o)
    
    # Fehlerberechnung Hidden Layer
    error_hidden = (W2.T @ error_out) * h * (1 - h)
    
    # Gewichtsaktualisierung (Gradient Descent)
    W2 -= learning_rate * outer(error_out, h)
    b2 -= learning_rate * error_out
    W1 -= learning_rate * outer(error_hidden, x)
    b1 -= learning_rate * error_hidden
```

**Ablauf des Lernprozesses**:

1. **Forward Pass**: Berechnung der Vorhersage
2. **Fehlerkalkulation**:
   - Erstellung eines Ziel-Vektors (One-Hot-Encoding)
   - Berechnung des Fehlers in der Output-Layer
   - Rückpropagierung des Fehlers zur Hidden-Layer

3. **Gradient Descent**:
   - Berechnung der Gradienten für alle Gewichte
   - Update der Gewichte in Richtung der Fehlerminimierung
   - Learning Rate kontrolliert die Schrittgröße

**Mathematische Grundlagen**:
- **Chain Rule** (Kettenregel): Zur Berechnung der Gradienten durch die Schichten
- **Gradient Descent**: Optimierungsalgorithmus zur Minimierung der Fehler
- **Sigmoid-Ableitung**: σ'(x) = σ(x) · (1 - σ(x))

### 4. Prediction (Vorhersage)

```python
def predict(self, x):
    _, o = self.forward(x)
    return argmax(o)
```

**Funktion**: Gibt den Index des Neurons mit der höchsten Aktivierung zurück, was der vorhergesagten Ziffer entspricht.

## Unterschiede: Implementation ohne NumPy vs. mit NumPy

### 1. Datenstrukturen

#### Ohne NumPy (net_old.py):
```python
# Verschachtelte Python-Listen
self.W1 = [[self.rand() for _ in range(self.input_size)]
           for _ in range(self.hidden_size)]
self.b1 = [0.0 for _ in range(self.hidden_size)]
```

#### Mit NumPy (net.py):
```python
# NumPy Arrays
self.W1 = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
self.b1 = np.zeros(hidden_size)
```

**Vorteile von NumPy**:
- Kompakterer, lesbarer Code
- Direkte Array-Initialisierung
- Optimierte Speicherverwaltung

### 2. Mathematische Operationen

#### Ohne NumPy (net_old.py):
```python
# Dot Product (Skalarprodukt) manuell
def dot(self, a, b):
    return sum(x * y for x, y in zip(a, b))

# Forward Pass mit Schleifen
h = [
    self.sigmoid(self.dot(self.W1[i], x) + self.b1[i])
    for i in range(self.hidden_size)
]

# Sigmoid mit math-Modul
def sigmoid(self, x):
    return 1 / (1 + math.exp(-x))
```

#### Mit NumPy (net.py):
```python
# Matrix-Vektor-Multiplikation mit @-Operator
h_raw = self.W1 @ x + self.b1
h = self.sigmoid(h_raw)

# Sigmoid mit NumPy (vektorisiert)
def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
```

**Vorteile von NumPy**:
- Vektorisierte Operationen (keine expliziten Schleifen)
- Schnellere Ausführung durch C-optimierte Bibliotheken
- Natürliche mathematische Notation (@-Operator für Matrixmultiplikation)

### 3. Backpropagation und Gewichtsaktualisierung

#### Ohne NumPy (net_old.py):
```python
# Fehlerberechnung Hidden Layer mit manuellen Schleifen
error_hidden = []
for i in range(self.hidden_size):
    s = sum(error_out[j] * self.W2[j][i] for j in range(self.output_size))
    error_hidden.append(s * h[i] * (1-h[i]))

# Gewichtsaktualisierung mit verschachtelten Schleifen
for i in range(self.output_size):
    for j in range(self.hidden_size):
        self.W2[i][j] -= self.learning_rate * error_out[i] * h[j]
    self.b2[i] -= self.learning_rate * error_out[i]
```

#### Mit NumPy (net.py):
```python
# Fehlerberechnung Hidden Layer vektorisiert
error_hidden = (self.W2.T @ error_out) * h * (1 - h)

# Gewichtsaktualisierung vektorisiert
self.W2 -= self.learning_rate * np.outer(error_out, h)
self.b2 -= self.learning_rate * error_out
```

**Vorteile von NumPy**:
- Eine Zeile statt verschachtelter Schleifen
- Deutlich bessere Performance
- Einfachere Fehlersuche und Wartung

### 4. Prediction-Funktion

#### Ohne NumPy (net_old.py):
```python
def predict(self, x):
    _, o = self.forward(x)
    return max(range(self.output_size), key=lambda i: o[i])
```

#### Mit NumPy (net.py):
```python
def predict(self, x):
    _, o = self.forward(x)
    return np.argmax(o)
```

**Vorteile von NumPy**:
- Eingebaute, optimierte `argmax`-Funktion
- Kürzerer, klarerer Code

### 5. Performance-Vergleich

| Aspekt | Ohne NumPy | Mit NumPy |
|--------|------------|-----------|
| **Geschwindigkeit** | Langsam (Python-Schleifen) | Schnell (C-optimiert) |
| **Speichereffizienz** | Weniger effizient | Sehr effizient |
| **Code-Länge** | Länger | Kürzer |
| **Lesbarkeit** | Mehr Boilerplate | Kompakter |
| **Training Zeit** | ~10x langsamer | Baseline |

**Geschwindigkeitsbeispiel**:
- Training von 60.000 Samples über 3 Epochen:
  - Ohne NumPy: ~15-20 Minuten
  - Mit NumPy: ~2-3 Minuten

### 6. Fehler in net_old.py

In der Implementierung ohne NumPy gibt es einen Fehler in den Zeilen 95-98:

```python
# FEHLER: Sollte hidden_size statt output_size verwenden
for i in range(self.output_size):  # ← FALSCH
    for j in range(self.hidden_size):
        self.W1[i][j] -= self.learning_rate * error_hidden[i] * x[j]
    self.b1[i] -= self.learning_rate * error_hidden[i]
```

**Korrektur**:
```python
for i in range(self.hidden_size):  # ← RICHTIG
    for j in range(self.input_size):
        self.W1[i][j] -= self.learning_rate * error_hidden[i] * x[j]
    self.b1[i] -= self.learning_rate * error_hidden[i]
```

Die NumPy-Version hat diesen Fehler nicht, da sie die richtigen Dimensionen durch `np.outer()` automatisch verwaltet.

## Training und Hyperparameter

### Verwendete Hyperparameter

```python
input_size = 784        # 28x28 Pixel
hidden_size = 30-170    # Variabel (experimentell)
output_size = 10        # 10 Ziffern (0-9)
learning_rate = 0.1-0.3 # Lernrate
epochs = 3              # Durchläufe durch den Datensatz
```

### Training Process

1. **Daten laden**: MNIST-Dataset (60.000 Trainingsbilder, 10.000 Testbilder)
2. **Normalisierung**: Pixelwerte werden durch 255 geteilt (0-1 Bereich)
3. **Training Loop**:
   - Für jede Epoche: Durchlaufen aller Trainingsbilder
   - Für jedes Bild: Forward Pass → Backpropagation → Gewichtsaktualisierung
4. **Evaluation**: Test auf separatem Testdatensatz

### Erreichte Genauigkeit

Typische Ergebnisse nach 3 Epochen:
- Hidden Size 30, Learning Rate 0.1: ~90-92% Genauigkeit
- Hidden Size 120-170, Learning Rate 0.2-0.3: ~93-95% Genauigkeit

## Zusammenfassung der Vorteile von NumPy

1. **Performance**: 5-10x schnellere Ausführung durch C-optimierte Operationen
2. **Code-Qualität**: Kürzerer, lesbarer Code ohne verschachtelte Schleifen
3. **Mathematische Notation**: Intuitive Matrix-Operationen (@-Operator)
4. **Fehlerreduktion**: Weniger Möglichkeiten für Index- und Dimensionsfehler
5. **Vektorisierung**: Automatische parallele Verarbeitung von Array-Operationen
6. **Broadcasting**: Automatische Anpassung von Array-Dimensionen
7. **Standard-Tool**: NumPy ist der De-facto-Standard für numerische Berechnungen in Python

## Dateistruktur des Projekts

```
NanoNeuron/
├── NNNet/
│   ├── net.py              # Optimierte NumPy-Implementation
│   ├── net_old.py          # Reine Python-Implementation
│   └── data_loader.py      # MNIST-Datenlader
├── data/                   # MNIST-Dataset
├── NNNet_saves/           # Gespeicherte trainierte Modelle
├── train_dif_configs.py   # Training mit verschiedenen Hyperparametern
├── test_with_testset.py   # Testen eines gespeicherten Modells
├── predict_random.py      # Vorhersage zufälliger Bilder
└── live_drawing_prediction.py  # Live-Zeichenerkennung
```

## Fazit

Das NanoNeuron-Projekt demonstriert eindrucksvoll die Vorteile von NumPy für maschinelles Lernen:

- Die **NumPy-Implementation** (net.py) ist kürzer, schneller und fehlerresistenter
- Die **reine Python-Implementation** (net_old.py) ist lehrreich, um die zugrunde liegenden Algorithmen zu verstehen
- Für produktive Anwendungen ist NumPy unverzichtbar aufgrund der massiven Performance-Vorteile

Das Projekt erreicht trotz seiner Einfachheit eine beachtliche Genauigkeit von ~93-95% bei der Erkennung handgeschriebener Ziffern und eignet sich hervorragend zum Verständnis der Grundlagen neuronaler Netzwerke.
