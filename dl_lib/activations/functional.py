import numpy as np


def sigmoid(arr: np.ndarray):
    return 1 / (1 + np.exp(-arr))


def sigmoid_derivative(arr: np.ndarray):
    return sigmoid(arr) * (1 - sigmoid(arr))


def relu(arr: np.ndarray):
    return np.maximum(arr, 0)


def tanh(arr: np.ndarray):
    return (np.exp(arr) - np.exp(-arr)) / (np.exp(arr) + np.exp(-arr))
