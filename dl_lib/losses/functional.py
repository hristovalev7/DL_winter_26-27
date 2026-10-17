import numpy as np


def mae(true_values, predicted_values):
    absolute_errors = np.abs(true_values - predicted_values)
    return np.mean(absolute_errors)
