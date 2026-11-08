import numpy as np


def initialize_weights(low, high, size, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size)


def forward(x, weights, bias=0.0, activation=None):
    if activation is not None:
        return activation(x @ weights + bias)
    return x @ weights + bias


def numerical_gradient(loss, parameters: np.ndarray) -> np.ndarray:
    epsilon = 1e-5
    gradient = np.zeros_like(parameters)
    for i in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus.flat[i] = plus.flat[i] + epsilon
        minus.flat[i] = minus.flat[i] - epsilon
        gradient.flat[i] = (loss(plus) - loss(minus)) / (2 * epsilon)
    return gradient


def gradient_descent_step(parameters: np.ndarray, gradient: np.ndarray, rate: float) -> np.ndarray:
    return parameters - rate * gradient
