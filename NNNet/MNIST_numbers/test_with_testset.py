import os.path

from NNNet.net import NNNet as NNNet
from NNNet import data_loader as dl
test_image_path = "data/numbers/t10k-images.idx3-ubyte"
test_label_path = "data/numbers/t10k-labels.idx1-ubyte"

print("load test data")
test_images = dl.load_images(test_image_path) / 255.0
test_labels = dl.load_labels(test_label_path)

nnnet = NNNet(input_size=784, hidden_size=30, output_size=10, seed=42, learning_rate=0.1)

nnnet.load_NNNet("number_net_saves/nnnet_save1_acc0.4936_hs30_lr0.1_seed42.npz")

correct = 0
for i in range(len(test_images)):
    if nnnet.predict(test_images[i]) == test_labels[i]:
        correct += 1
accuracy = f"{correct / len(test_images):.4f}"
print(f"Final Test Accuracy: {accuracy}")