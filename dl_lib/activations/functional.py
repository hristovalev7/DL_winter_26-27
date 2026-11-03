import numpy as np


def sigmoid_helper(x):
    return 1 / (1 + np.pow(np.e, -x))


def sigmoid(arr: np.ndarray):
    return sigmoid_helper(arr)


def sigmoid_derivative(arr: np.ndarray):
    return sigmoid(arr) * (1 - sigmoid(arr))


def relu(arr: np.ndarray):
    return np.maximum(arr, 0)


def tanh(arr: np.ndarray):
    return (np.exp(arr) - np.exp(-arr)) / (np.exp(arr) + np.exp(-arr))
