from NNNet import net
from NNNet.net import NNNet
from training_data.letters.csv_dataset import CSVDataset


def train_letters_NNNet(
        input_size=784,
        hidden_size=300,
        output_size=26,
        seed=42,
        learning_rate=0.2,
        debug=False,
):
    # Dataset laden
    dataset = CSVDataset('C:\\Users\\fridi\\PycharmProjects\\NanoNeuron\\data\\letters\\emnist-letters-train.csv\\emnist-letters-train.csv')

    # Einzelnes Sample
    image, label = dataset.get_sample(0)

    nnnet = NNNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, seed=seed, learning_rate=learning_rate)

    EPOCHS = 5
    TRAIN_SAMPLES = len(dataset)

    # Training
    for epoch in range(EPOCHS):

        correct = 0

        for i in range(TRAIN_SAMPLES):
            image, label = dataset[i]
            nnnet.train(image, label, False)

            pred = nnnet.predict(image)
            if pred == label:
                correct += 1

        print(f"EPOCH: {epoch + 1}    ---    Training Accuracy: {correct / TRAIN_SAMPLES:.2f}")


    nnnet.save_NNNet(f"C:\\Users\\fridi\\PycharmProjects\\NanoNeuron\\NNNet_saves_ExtendedMNIST\\Letters\\LetterNet_hs{hidden_size}_os{output_size}_seed{seed}_lr{learning_rate}")

if __name__ == "__main__":
    train_letters_NNNet()