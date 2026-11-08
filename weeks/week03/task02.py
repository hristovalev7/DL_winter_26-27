import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import initialize_weights, forward


def main():
    dataset_list = create_dataset(6)
    dataset_array = np.array(dataset_list)
    x = dataset_array[:, 0]
    y = dataset_array[:, 1]
    x_column = x.reshape(-1, 1)
    weight = initialize_weights(0, 10, 1, 42)
    predictions = forward(x_column, weight)
    print(f"Dataset: {dataset_list}")
    print(f"w: {weight[0]}")
    print(f"MSE: {mse(y, predictions)}")
    print("Perturbing w:")
    epsilons = np.arange(0.002, step=-0.001, stop=-0.003)
    for e in epsilons:
        if np.isclose(e, 0, 1e-7):
            continue
        predictions = forward(x_column, weight + e)
        print(f"w + ({e}) -> w={weight[0] + e}, MSE={mse(y, predictions)}")
