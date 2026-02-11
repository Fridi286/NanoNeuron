from NNNet.MNIST_numbers.train.train_NNNET import train_NNNet


def train_configs():
    # -------------------------------
    # Konfigurationsräume
    # -------------------------------
    learning_rates = [
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
    ]

    hidden_layer_configs = [
        # 1 Hidden Layer (Breite)
        [16],
        [32],
        [64],
        [128],
        [256],

        # 2 Hidden Layers
        [32, 32],
        [64, 64],
        [128, 64],
        [128, 128],
        [256, 128],

        # 3 Hidden Layers
        [64, 64, 64],
        [128, 64, 32],
        [128, 128, 64],
        [256, 128, 64],

        # 4 Hidden Layers
        [64, 64, 64, 64],
        [128, 128, 64, 32],
    ]

    batch_sizes = [32, 64, 128]

    seeds = [42]

    EPOCHS = 40

    # -------------------------------
    # Grid-Search Schleife
    # -------------------------------

    for seed in seeds:
        for hidden_layers in hidden_layer_configs:
            for lr in learning_rates:

                # ---------- NO BATCH ----------
                # Sigmoid
                train_NNNet(
                    input_size=784,
                    hidden_layers=hidden_layers,
                    output_size=10,
                    seed=seed,
                    learning_rate=lr,
                    pre_trained=None,
                    relu=False,
                    use_batches=False,
                    batch_size=1,
                    use_gpu=False,
                    EPOCHS=EPOCHS,
                )

                # ReLU
                train_NNNet(
                    input_size=784,
                    hidden_layers=hidden_layers,
                    output_size=10,
                    seed=seed,
                    learning_rate=lr,
                    pre_trained=None,
                    relu=True,
                    use_batches=False,
                    batch_size=1,
                    use_gpu=False,
                    EPOCHS=EPOCHS,
                )

                # ---------- WITH BATCH ----------
                for batch_size in batch_sizes:
                    # Sigmoid + Batch
                    train_NNNet(
                        input_size=784,
                        hidden_layers=hidden_layers,
                        output_size=10,
                        seed=seed,
                        learning_rate=lr,
                        pre_trained=None,
                        relu=False,
                        use_batches=True,
                        batch_size=batch_size,
                        use_gpu=False,
                        EPOCHS=EPOCHS,
                    )

                    # ReLU + Batch
                    train_NNNet(
                        input_size=784,
                        hidden_layers=hidden_layers,
                        output_size=10,
                        seed=seed,
                        learning_rate=lr,
                        pre_trained=None,
                        relu=True,
                        use_batches=True,
                        batch_size=batch_size,
                        use_gpu=False,
                        EPOCHS=EPOCHS,
                    )

                    # ---------- OPTIONAL: GPU ----------
                    """
                    use_gpu = True
                    train_NNNet(
                        input_size=784,
                        hidden_layers=hidden_layers,
                        output_size=10,
                        seed=seed,
                        learning_rate=lr,
                        pre_trained=None,
                        relu=True,
                        use_batches=True,
                        batch_size=batch_size,
                        use_gpu=True,
                        EPOCHS=EPOCHS,
                    )
                    """

if __name__ == "__main__":
    train_configs()