import tkinter as tk
from tkinter import messagebox

import numpy as np

from NNNet.net import NNNet  # ggf. an dein Projekt anpassen


# ------------------- NNNet laden ------------------- #

def load_model():
    nn = NNNet(input_size=784, hidden_size=65, seed=52, learning_rate=0.2, output_size=10)
    nn.load_NNNet("NNNet_saves/nnnet_save1_acc0.7590_hs65_lr0.2_seed52.npz")
    return nn


# ------------------- Zeichen-Tool ------------------- #

class DigitDrawApp:
    def __init__(self, master, nnnet):
        self.master = master
        self.master.title("NNNet Digit Predictor")

        self.nnnet = nnnet

        # Logische Größe des Bildes (MNIST: 28x28)
        self.grid_size = 28
        # Anzeigegröße in Pixeln
        self.canvas_size = 280
        self.cell_size = self.canvas_size // self.grid_size

        # internes 28x28 Bild (0 = schwarz, 1 = weiß)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # dünner, scharfer, nicht-weicher weißer Pinsel
        self.kernel = np.array([
            [1.0, 1.0],
            [1.0, 1.0]
        ], dtype=np.float32)

        # Canvas
        self.canvas = tk.Canvas(
            master,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",          # Hintergrund schwarz
            highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=10)

        # Rechtecke für 28×28 Pixel
        self.rects = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                x0 = gx * self.cell_size
                y0 = gy * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                rect_id = self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill="#000000",        # schwarz
                    outline="#303030"      # dezente Grid-Linie
                )
                self.rects[gy][gx] = rect_id

        # Buttons
        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=5)

        self.predict_button = tk.Button(btn_frame, text="Predict", command=self.predict_digit)
        self.predict_button.grid(row=0, column=0, padx=5)

        self.clear_button = tk.Button(btn_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=0, column=1, padx=5)

        # Label für Ausgabe
        self.result_var = tk.StringVar()
        self.result_var.set("Draw a digit with WHITE on BLACK")
        self.result_label = tk.Label(master, textvariable=self.result_var, font=("Arial", 14))
        self.result_label.pack(pady=5)

        # Maus-Events
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.last_cell = None

    # --------- Zeichnen auf dem 28x28 Grid ---------- #

    def on_button_press(self, event):
        self.last_cell = None
        self._paint_at_event(event)

    def on_paint(self, event):
        self._paint_at_event(event)

    def on_button_release(self, event):
        self.last_cell = None

    def _paint_at_event(self, event):
        gx = event.x // self.cell_size
        gy = event.y // self.cell_size

        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            self.apply_brush(gx, gy)
            self.update_canvas_from_grid()

    def apply_brush(self, cx, cy):
        """
        Harte Kante, dünner weißer Pinsel (Kernel = [[1.0]]).
        """
        kh, kw = self.kernel.shape
        off_y = kh // 2
        off_x = kw // 2

        for ky in range(kh):
            for kx in range(kw):
                gx = cx + (kx - off_x)
                gy = cy + (ky - off_y)
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    self.grid[gy, gx] = np.clip(
                        self.grid[gy, gx] + self.kernel[ky, kx],
                        0.0,
                        1.0
                    )

    def update_canvas_from_grid(self):
        """
        grid[gy, gx] (0..1) -> Farbe im Rechteck.
        0 = schwarz (#000000), 1 = weiß (#FFFFFF)
        """
        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                intensity = float(self.grid[gy, gx])  # 0..1
                c = int(intensity * 255)
                color = f"#{c:02x}{c:02x}{c:02x}"
                self.canvas.itemconfig(self.rects[gy][gx], fill=color)

    # --------- Utilities ---------- #

    def clear_canvas(self):
        self.grid.fill(0.0)
        self.update_canvas_from_grid()
        self.result_var.set("Draw a digit with WHITE on BLACK")

    def preprocess_image(self):
        """
        Gibt direkt das 28x28-Grid als Vektor (784,) zurück.
        Werte: 0..1
        """
        x = self.grid.reshape(self.grid_size * self.grid_size).astype(np.float32)
        return x

    def predict_digit(self):
        x = self.preprocess_image()
        try:
            pred, o = self.nnnet.predict_debug(x)
        except Exception as e:
            messagebox.showerror("Error", f"Fehler bei nnnet.predict(x):\n{e}")
            return

        self.result_var.set(f"Prediction: {pred}")


def main():
    nnnet = load_model()
    root = tk.Tk()
    app = DigitDrawApp(root, nnnet)
    root.mainloop()


if __name__ == "__main__":
    main()
