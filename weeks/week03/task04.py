import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import forward, initialize_weights, analytical_gradient, numerical_gradient


def loss_function(weights: np.ndarray) -> np.ndarray:
    dataset = np.array(create_dataset(6))
    x = dataset[:, 0].reshape(-1, 1)
    true_values = dataset[:, 1]
    predicted_values = forward(x, weights)
    return mse(true_values, predicted_values)


def main():
    dataset = np.array(create_dataset(6))
    weights = initialize_weights(0, 10, 1, 42)
    a_gradient = analytical_gradient(dataset, weights)[0]
    n_gradient = numerical_gradient(loss_function, weights)[0]
    print(f"Loss: {loss_function(weights)}")
    print(f"Numerical gradient: {n_gradient:.4f}")
    print(f"Analytical gradient: {a_gradient:.4f}")
    print(f"Difference: {(a_gradient - n_gradient):.4f}")
