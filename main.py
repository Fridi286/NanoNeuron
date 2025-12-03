import net
from net import NNNet as NNNet
import data_loader as dl

train_image_path = "data/train-images.idx3-ubyte"
train_label_path = "data/train-labels.idx1-ubyte"

test_image_path = "data/t10k-images.idx3-ubyte"
test_label_path = "data/t10k-labels.idx1-ubyte"

print("load train data")
train_images = dl.load_images(train_image_path) / 255.0
train_labels = dl.load_labels(train_label_path)

print("load test data")
test_images = dl.load_images(test_image_path) / 255.0
test_labels = dl.load_labels(test_label_path)

print("Create neuronal network NNNet")
nnnet = NNNet(input_size=784, hidden_size=16, output_size=10, seed=10, learning_rate=0.1)

print("initializing training loop")
EPOCHS = 3
TRAIN_SAMPLES = 5000

print("Start Training...")

for epoch in range(EPOCHS):
    correct = 0

    for i in range(TRAIN_SAMPLES):
        x = train_images[i]
        label = train_labels[i]

        nnnet.train(x, label)

        #Testing testwise while running
        pred = nnnet.predict(x)
        if pred == label:
            correct += 1

        #every 500 samples output current accuracy
        if i % 500 == 0:
            print(f"EPOCH: {epoch+1}    ---    Training Accuracy: {correct/(i+1):.2f}")

    # accuracry after each epoch
    print(f"EPOCH: {epoch+1}    ---    Training Accuracy: {correct/TRAIN_SAMPLES:.2f}")

print("Training finished")

print("Start Testing")

correct = 0
for i in range(len(test_images)):
    if nnnet.predict(test_images[i]) == test_labels[i]:
        correct += 1

print(f"Final Test Accuracy: {correct / len(test_images):.4f}")
