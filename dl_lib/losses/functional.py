import numpy as np


def mae(true_values, predicted_values):
    absolute_errors = np.abs(true_values - predicted_values)
    return np.mean(absolute_errors)


def mse(true_values, predicted_values):
    squared_errors_sum = (true_values - predicted_values) @ (
        true_values - predicted_values
    )
    return squared_errors_sum / len(predicted_values)


def rmse(true_values, predicted_values):
    return np.sqrt(mse(true_values, predicted_values))
