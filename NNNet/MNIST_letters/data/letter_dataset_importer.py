import numpy as np
import pandas as pd

# Klasse lädt CSV Dataset von mnist statt die ubytes
class CSVDataset:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

        # Labels (erste Spalte)
        self.labels = self.data.iloc[:, 0].values - 1

        # Pixelwerte (alle Spalten nach der ersten)
        self.images = self.data.iloc[:, 1:].values

        # Normalisierung der Pixelwerte auf [0, 1]
        self.images = self.images / 255.0

        # EMNIST-Letters Korrektur: 90° rechts drehen + horizontal spiegeln
        corrected_images = []
        for img in self.images:
            img_2d = img.reshape(28, 28)
            img_2d = np.rot90(img_2d, k=-1)  # 90° nach rechts drehen
            img_2d = np.fliplr(img_2d)  # horizontal spiegeln
            corrected_images.append(img_2d.flatten())

        self.images = np.array(corrected_images)

    # Single sample predicten
    def get_sample(self, index):
        return self.images[index], self.labels[index]

    # Batch sample predicten
    def get_batch(self, batch_size):
        indices = np.random.choice(len(self.images), batch_size, replace=False)
        return self.images[indices], self.labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.get_sample(index)
