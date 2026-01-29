from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from pydantic import BaseModel
import matplotlib.pyplot as plt
import cv2

from NNNet.net import NNNet


# -----------------------------
# 1) Config
# -----------------------------

class GAConfig(BaseModel):
    image_shape: Tuple[int, int] = (28, 28)
    population_size: int = 200
    generations: int = 300
    elite_fraction: float = 0.05
    tournament_k: int = 5
    mutation_rate: float = 0.3  # Höher für Striche
    crossover_rate: float = 0.5
    seed: Optional[int] = 42
    num_strokes: int = 8  # Anzahl der Liniensegmente
    stroke_thickness: int = 2


# -----------------------------
# 2) Individual (Genom = Striche)
# -----------------------------

@dataclass
class Individual:
    # Genom: (num_strokes, 4) -> jeder Strich: [x1, y1, x2, y2] normalisiert [0,1]
    genome: np.ndarray
    fitness: float = float("-inf")

    def copy(self) -> "Individual":
        return Individual(self.genome.copy(), self.fitness)

    def render(self, img_shape: Tuple[int, int], thickness: int) -> np.ndarray:
        """Zeichnet die Striche auf ein leeres Bild"""
        H, W = img_shape
        img = np.zeros((H, W), dtype=np.uint8)

        for stroke in self.genome:
            x1 = int(stroke[0] * (W - 1))
            y1 = int(stroke[1] * (H - 1))
            x2 = int(stroke[2] * (W - 1))
            y2 = int(stroke[3] * (H - 1))

            cv2.line(img, (x1, y1), (x2, y2), color=1, thickness=thickness)

        return img.astype(np.float32)


# -----------------------------
# 3) Fitness
# -----------------------------

def center_of_mass_penalty(img: np.ndarray) -> float:
    """Bestraft Bilder, deren Schwerpunkt nicht mittig ist"""
    H, W = img.shape
    y_coords, x_coords = np.nonzero(img)

    if len(x_coords) == 0:
        return 10.0  # Kein Pixel gesetzt

    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)

    target_x = W / 2
    target_y = H / 2

    dist = np.sqrt((center_x - target_x) ** 2 + (center_y - target_y) ** 2)
    return dist / W  # Normalisiert


class NeuralNetFitness:
    def __init__(self):
        self.nn = NNNet(input_size=784, hidden_size=120, output_size=10, seed=1000, learning_rate=0.25)
        self.nn.load_NNNet(
            r"C:\Users\fridi\PycharmProjects\NanoNeuron\NNNet_saves\59999samples\nnnet_save1_acc0.9710_hs120_lr0.25_seed1000.npz")

    def evaluate(self, images: np.ndarray, target_class: int) -> np.ndarray:
        N = images.shape[0]
        fitness = np.zeros(N, dtype=np.float32)

        for i in range(N):
            img = images[i]
            x = img.flatten()

            _, o = self.nn.forward(x)
            o = np.asarray(o).reshape(-1)

            target_score = float(o[target_class])
            other_max = float(np.max(np.delete(o, target_class)))

            # Hauptziel: Klare Klassifikation
            classification_score = target_score - other_max

            # Priors für MNIST-ähnliche Bilder
            ink = float(np.mean(img))  # Sollte zwischen 0.1-0.3 liegen
            ink_penalty = abs(ink - 0.15) * 2.0  # Bestraft zu viel/wenig Tinte

            center_penalty = center_of_mass_penalty(img)

            edge_pixels = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
            edge_penalty = float(np.mean(edge_pixels))

            # Kompaktheit: Bestraft zu verstreute Pixel
            if ink > 0:
                y_coords, x_coords = np.nonzero(img)
                if len(x_coords) > 1:
                    spread = np.std(x_coords) + np.std(y_coords)
                    compactness_penalty = spread / 14.0  # Normalisiert
                else:
                    compactness_penalty = 5.0
            else:
                compactness_penalty = 5.0

            fitness[i] = (
                    5.0 * classification_score  # Hauptgewicht auf Klassifikation
                    - 0.5 * ink_penalty  # Realistische Tintenmenge
                    - 0.4 * center_penalty  # Mittig positioniert
                    - 0.6 * edge_penalty  # Rand dunkel
                    - 0.3 * compactness_penalty  # Kompakt, nicht verstreut
            )

        return fitness


# -----------------------------
# 4) GA Engine
# -----------------------------

class GeneticImageOptimizer:
    def __init__(self, config: GAConfig, fitness: NeuralNetFitness):
        self.cfg = config
        self.fitness = fitness
        self.rng = np.random.default_rng(config.seed)

    def run(self, target_class: int) -> Individual:
        population = self._init_population()

        for gen in range(self.cfg.generations):
            self._evaluate_population(population, target_class)
            population.sort(key=lambda ind: ind.fitness, reverse=True)

            best = population[0]
            print(f"Gen {gen:4d} | best fitness = {best.fitness:.6f}")

            population = self._next_generation(population)

        self._evaluate_population(population, target_class)
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        return population[0]

    def _init_population(self) -> List[Individual]:
        pop = []
        for _ in range(self.cfg.population_size):
            # Zufällige Striche: (num_strokes, 4) mit Werten in [0,1]
            strokes = self.rng.random(size=(self.cfg.num_strokes, 4)).astype(np.float32)
            pop.append(Individual(strokes))
        return pop

    def _evaluate_population(self, population: List[Individual], target_class: int) -> None:
        imgs = np.stack([
            ind.render(self.cfg.image_shape, self.cfg.stroke_thickness)
            for ind in population
        ], axis=0)

        fit = self.fitness.evaluate(imgs, target_class)
        for ind, f in zip(population, fit):
            ind.fitness = float(f)

    def _next_generation(self, population: List[Individual]) -> List[Individual]:
        N = self.cfg.population_size
        elite_n = max(1, int(N * self.cfg.elite_fraction))

        next_pop = [population[i].copy() for i in range(elite_n)]

        while len(next_pop) < N:
            parent_a = self._tournament_select(population)
            parent_b = self._tournament_select(population)

            child = self._make_child(parent_a, parent_b)
            self._mutate(child)

            next_pop.append(child)

        return next_pop

    def _tournament_select(self, population: List[Individual]) -> Individual:
        k = self.cfg.tournament_k
        idx = self.rng.integers(0, len(population), size=k)
        candidates = [population[i] for i in idx]
        return max(candidates, key=lambda ind: ind.fitness)

    def _make_child(self, a: Individual, b: Individual) -> Individual:
        if self.rng.random() > self.cfg.crossover_rate:
            return Individual(a.genome.copy())

        # Crossover: Mische Striche
        mask = self.rng.random(size=a.genome.shape[0]) < 0.5
        child_genome = np.where(mask[:, None], a.genome, b.genome).astype(np.float32)
        return Individual(child_genome)

    def _mutate(self, ind: Individual) -> None:
        # Mutiere zufällige Striche
        num_strokes = ind.genome.shape[0]
        num_mutations = max(1, int(num_strokes * self.cfg.mutation_rate))

        stroke_indices = self.rng.choice(num_strokes, size=num_mutations, replace=False)

        for idx in stroke_indices:
            # Zufällige Änderung eines Strichs
            if self.rng.random() < 0.5:
                # Kleine Änderung
                ind.genome[idx] += self.rng.normal(0, 0.1, size=4)
            else:
                # Komplett neuer Strich
                ind.genome[idx] = self.rng.random(size=4)

            # Clipping auf [0,1]
            ind.genome[idx] = np.clip(ind.genome[idx], 0, 1)


# -----------------------------
# 5) Visualisierung
# -----------------------------

def visualize_result(individual: Individual, target_class: int, fitness_obj: NeuralNetFitness, cfg: GAConfig):
    """Zeigt das generierte Bild und die NN-Ausgabe"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bild rendern und anzeigen
    img = individual.render(cfg.image_shape, cfg.stroke_thickness)
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'Generiertes Bild für Ziffer {target_class}')
    ax1.axis('off')

    # NN-Ausgabe
    x_flat = img.flatten()
    _, o = fitness_obj.nn.forward(x_flat)

    ax2.bar(range(10), o, color=['green' if i == target_class else 'blue' for i in range(10)])
    ax2.set_xlabel('Ziffer')
    ax2.set_ylabel('Aktivierung')
    ax2.set_title(f'NN-Ausgabe (Fitness: {individual.fitness:.4f})')
    ax2.set_xticks(range(10))
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


# -----------------------------
# 6) Hauptprogramm
# -----------------------------

if __name__ == "__main__":
    cfg = GAConfig(
        population_size=500,  # Mehr Diversität
        generations=10000,  # Mehr Zeit
        num_strokes=20,  # Mehr Striche für komplexere Formen
        stroke_thickness=3,  # Dickere Linien
        mutation_rate=0.4  # Weniger aggressive Mutation
    )
    fitness = NeuralNetFitness()
    ga = GeneticImageOptimizer(cfg, fitness)

    target_digit = 0
    best = ga.run(target_class=target_digit)
    print(f"\nBest fitness: {best.fitness:.6f}")

    visualize_result(best, target_digit, fitness, cfg)
