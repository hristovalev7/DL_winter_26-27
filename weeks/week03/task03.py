import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import initialize_weights, forward, numerical_gradient, gradient_descent_step


def loss_function(weights: np.ndarray) -> np.ndarray:
    dataset = np.array(create_dataset(6))
    inputs = dataset[:, 0]
    inputs_column = inputs.reshape(-1, 1)
    true_values = dataset[:, 1]
    predicted_values = forward(inputs_column, weights)
    return mse(true_values, predicted_values)


def main():
    weights = initialize_weights(0, 10, 1, 42)
    learning_rate = 0.02867
    loss = loss_function(weights)
    gradient = numerical_gradient(loss_function, weights)
    print(f"Loss before update: {loss}")
    print(f"Numerical gradient: {gradient}")
    print(f"w after subtracting the raw gradient: w={(weights[0] - gradient)[0]}, loss={loss_function(weights - gradient)}")
    print(f"w after one step with learning_rate=0.001: w={weights[0]}, loss={loss_function(gradient_descent_step(weights, gradient, 0.001))}")

    print(f"Loss before update: {loss}")
    print()
    for i in range(10):
        gradient = numerical_gradient(loss_function, weights)
        weights = gradient_descent_step(weights, gradient, learning_rate)
        print(f"epoch {i}: w={weights - learning_rate * gradient}, loss={loss_function(weights)}")
