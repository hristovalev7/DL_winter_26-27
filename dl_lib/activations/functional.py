import numpy as np


def sigmoid(arr: np.ndarray):
    return 1 / (1 + np.exp(-arr))


def sigmoid_derivative(arr: np.ndarray):
    return sigmoid(arr) * (1 - sigmoid(arr))


def relu(arr: np.ndarray):
    return np.maximum(arr, 0)


def tanh(arr: np.ndarray):
    return (np.exp(arr) - np.exp(-arr)) / (np.exp(arr) + np.exp(-arr))


def softmax(arr: np.ndarray):
    m = np.max(arr, axis=-1, keepdims=True)
    return np.exp(arr - m) / np.sum(np.exp(arr - m), axis=-1, keepdims=True)


def naive_softmax(arr: np.ndarray):
    return np.exp(arr) / np.sum(np.exp(arr), axis=-1)
