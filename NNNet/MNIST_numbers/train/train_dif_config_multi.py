from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from NNNet.MNIST_numbers.train.train_NNNET import train_NNNet


def make_jobs(seeds, hidden_layer_configs, learning_rates, batch_sizes, epochs):
    jobs = []
    for seed in seeds:
        for hidden_layers in hidden_layer_configs:
            for lr in learning_rates:
                # NO BATCH: Sigmoid + ReLU
                jobs.append(dict(hidden_layers=hidden_layers, seed=seed, lr=lr,
                                relu=False, use_batches=False, batch_size=1, use_gpu=False, epochs=epochs))
                jobs.append(dict(hidden_layers=hidden_layers, seed=seed, lr=lr,
                                relu=True, use_batches=False, batch_size=1, use_gpu=False, epochs=epochs))

                # WITH BATCH
                for bs in batch_sizes:
                    jobs.append(dict(hidden_layers=hidden_layers, seed=seed, lr=lr,
                                    relu=False, use_batches=True, batch_size=bs, use_gpu=False, epochs=epochs))
                    jobs.append(dict(hidden_layers=hidden_layers, seed=seed, lr=lr,
                                    relu=True, use_batches=True, batch_size=bs, use_gpu=False, epochs=epochs))
    return jobs


def run_one_job(job):
    # Hier drin läuft EIN Trainingslauf in einem eigenen Prozess
    train_NNNet(
        input_size=784,
        hidden_layers=job["hidden_layers"],
        output_size=10,
        seed=job["seed"],
        learning_rate=job["lr"],
        pre_trained=None,
        relu=job["relu"],
        use_batches=job["use_batches"],
        batch_size=job["batch_size"],
        use_gpu=job["use_gpu"],
        EPOCHS=job["epochs"],
    )
    return job  # optional: zurückgeben, was fertig ist


def train_configs_parallel(max_workers=None):
    learning_rates = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.05, 0.1, 0.15, 0.2]
    hidden_layer_configs = [
        [16], [32], [64], [128], [256],
        [32, 32], [64, 64], [128, 64], [128, 128], [256, 128],
        [64, 64, 64], [128, 64, 32], [128, 128, 64], [256, 128, 64],
        [64, 64, 64, 64], [128, 128, 64, 32],
    ]
    batch_sizes = [32, 64, 128]
    seeds = [42]
    EPOCHS = 50

    jobs = make_jobs(seeds, hidden_layer_configs, learning_rates, batch_sizes, EPOCHS)

    # Empfehlung: nicht zu viele Worker, sonst RAM + IO Overhead
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) // 2)

    print(f"Total jobs: {len(jobs)} | max_workers={max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one_job, job) for job in jobs]
        for f in as_completed(futures):
            job = f.result()
            print(f"Done: seed={job['seed']} HL={job['hidden_layers']} lr={job['lr']} "
                  f"relu={job['relu']} batches={job['use_batches']} bs={job['batch_size']} gpu={job['use_gpu']}")


if __name__ == "__main__":
    train_configs_parallel(max_workers=6)  # z.B. 4
