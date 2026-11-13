import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import forward, initialize_weights


def main():
    dataset = create_dataset(6)
    x = np.array([pair[0] for pair in dataset]).reshape(-1, 1)
    y = np.array([pair[1] for pair in dataset])
    weight = initialize_weights(0, 10, 1, 42)
    predictions = forward(x, weight)
    print(f"Dataset: {dataset}")
    print(f"w: {weight[0]}")
    print(f"MSE: {mse(y, predictions)}")
    print("Perturbing w:")
    epsilons = np.arange(0.002, step=-0.001, stop=-0.003)
    for e in epsilons:
        if np.isclose(e, 0, 1e-7):
            continue
        predictions = forward(x, weight + e)
        print(f"w + ({e}) -> w={weight[0] + e}, MSE={mse(y, predictions)}")
