import net
from net import NNNet as NNNet
import data_loader as dl

train_image_path = "data/train-images.idx3-ubyte"
train_label_path = "data/train-labels.idx1-ubyte"


train_images = dl.load_images(train_image_path) / 255   # normalization
train_labels = dl.load_labels(train_label_path)


nnnet = NNNet(input_size=784, hidden_size=16, output_size=10, seed=10)

# Ein Bild ist ein Vektor der Länge 784 (normalisiert 0..1)
x = train_images[5]          # z.B. erstes Bild
label = train_labels[5]      # z.B. die echte Zahl

nnnet.train(x, label)          # einen Trainingsschritt

pred = nnnet.predict(x)        # Vorhersage
print(pred, label)