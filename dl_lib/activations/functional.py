import numpy as np


def sigmoid_helper(x):
    return 1 / (1 + np.pow(np.e, -x))


def sigmoid(array: np.ndarray):
    return sigmoid_helper(array)
