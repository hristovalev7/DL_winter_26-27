import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import (
    analytical_gradient,
    forward,
    initialize_weights,
)


def main():
    learning_rate = 0.001
    dataset = np.array(create_dataset(6))
    weights = initialize_weights(0, 10, 1, 42)
    print(f"Seed 42, initial w={weights[0]}")
    x = dataset[:, 0].reshape(-1, 1)
    y_true = dataset[:, 1]
    current_loss = np.nan
    for epoch in range(500):
        (w_gradient, _) = analytical_gradient(dataset, weights)
        weights = weights - learning_rate * w_gradient

        y_pred = forward(x, weights)
        current_loss = mse(y_true, y_pred)
    print(
        f"After 500 epochs: w={weights[0]:.6f}, loss={current_loss:.6f} (target: w=2.0)"
    )

    print()

    print("Training again without a seed")
    weights = initialize_weights(0, 10, 1)
    print(f"No seed, initial w={weights[0]}")
    current_loss = np.nan
    for epoch in range(500):
        (w_gradient, _) = analytical_gradient(dataset, weights)
        weights = weights - learning_rate * w_gradient

        y_pred = forward(x, weights)
        current_loss = mse(y_true, y_pred)
    print(
        f"After 500 epochs: w={weights[0]:.6f}, loss={current_loss:.6f} (target: w=2.0)"
    )
