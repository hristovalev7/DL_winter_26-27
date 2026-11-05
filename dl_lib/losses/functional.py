import numpy as np
from numpy import clip


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


def binary_cross_entropy(true_labels, predicted_labels):
    epsilon = 1e-7
    clipped_probabilities = clip(predicted_labels, epsilon, 1 - epsilon)
    return -(np.mean(true_labels * np.log(clipped_probabilities) + (1 - true_labels) * np.log(1 - clipped_probabilities)))


def categorical_cross_entropy(true_labels, predicted_probabilities):
    epsilon = 1e-7
    clipped_probabilities = clip(predicted_probabilities, epsilon, 1 - epsilon)
    return -(np.mean(np.log(clipped_probabilities[np.arange(len(true_labels)), true_labels])))
