import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import threading
import time

# Matplotlib Einbettung in Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ------------------------------------------
# Dein Modell laden
# ------------------------------------------
from NNNet.net import NNNet

def load_model():
    nn = NNNet(input_size=784, hidden_size=120, seed=52, learning_rate=0.2, output_size=10)
    nn.load_NNNet("NNNet_saves/5000samples/nnnet_save1_acc0.8238_hs120_lr0.2_seed52.npz")
    return nn

nnnet = load_model()


# ===================================================================
# UI – Zeichenfläche + Live-Bar-Chart + MNIST-Vorschau
# ===================================================================

CANVAS_SIZE = 280
MNIST_SIZE = 28
UPDATE_INTERVAL = 0.05


class MNISTDrawer:
    def __init__(self, root):
        self.root = root
        root.title("Live MNIST Predictor")

        # ----------------------------------------
        # Canvas zum Zeichnen
        # ----------------------------------------
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, rowspan=2, padx=10, pady=10)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

        self.canvas.bind("<ButtonPress-1>", self.set_last_pos)
        self.canvas.bind("<B1-Motion>", self.paint)

        tk.Button(root, text="Löschen", command=self.clear).grid(row=2, column=0, pady=5)

        # ----------------------------------------
        # Bar Chart
        # ----------------------------------------
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.ax.set_title("Prediction Probabilities")
        self.ax.set_ylim([0, 1])
        self.ax.set_xticks(range(10))
        self.ax.set_xticklabels([str(i) for i in range(10)])

        self.bars = self.ax.bar(range(10), [0]*10)

        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.chart_canvas.get_tk_widget().grid(row=0, column=1, padx=20, pady=10)

        # ----------------------------------------
        # MNIST Bildvorschau (28×28 hochskaliert)
        # ----------------------------------------
        self.preview_label = tk.Label(root)
        self.preview_label.grid(row=1, column=1, pady=10)

        # Prediction Thread
        self.running = True
        threading.Thread(target=self.update_loop, daemon=True).start()

    # ----------------------------------------
    # Zeichnen
    # ----------------------------------------
    def set_last_pos(self, event):
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=20, fill="black", capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                           width=20, fill=0)
        self.last_x, self.last_y = event.x, event.y

    # ----------------------------------------
    # Canvas löschen
    # ----------------------------------------
    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)

    # ----------------------------------------
    # MNIST-Format erzeugen
    # ----------------------------------------
    def get_mnist_image(self):
        img = self.image.copy()
        img = img.resize((MNIST_SIZE, MNIST_SIZE))
        img = ImageOps.invert(img)

        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.reshape(784)

        return arr, img  # arr fürs NN, img für Preview

    # ----------------------------------------
    # Loop für Prediction
    # ----------------------------------------
    def update_loop(self):
        while self.running:
            time.sleep(UPDATE_INTERVAL)

            x, preview_img = self.get_mnist_image()
            pred, o = nnnet.predict_debug(x)

            self.root.after(0, lambda: self.update_chart(o))
            self.root.after(0, lambda: self.update_preview(preview_img))

    # ----------------------------------------
    # Balkendiagramm aktualisieren
    # ----------------------------------------
    def update_chart(self, probs):
        for i, bar in enumerate(self.bars):
            bar.set_height(probs[i])

        self.ax.set_ylim([0, max(1, max(probs) + 0.05)])
        self.chart_canvas.draw_idle()

    # ----------------------------------------
    # MNIST-Bildvorschau aktualisieren
    # ----------------------------------------
    def update_preview(self, mnist_img):
        # MNIST 28×28 → sichtbar machen (z.B. 200×200)
        img_big = mnist_img.resize((200, 200), Image.NEAREST)

        self.preview = ImageTk.PhotoImage(img_big)
        self.preview_label.config(image=self.preview)


# ===================================================================
# Start
# ===================================================================
from PIL import ImageTk  # wichtig für Bildanzeige
root = tk.Tk()
MNISTDrawer(root)
root.mainloop()
