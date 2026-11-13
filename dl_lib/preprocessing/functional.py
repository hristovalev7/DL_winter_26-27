import numpy as np


def min_max_normalize(array):
    min_element = np.min(array)
    max_element = np.max(array)
    array_shifted_by_min = array - min_element
    if max_element == min_element:
        return array_shifted_by_min
    return array_shifted_by_min / (max_element - min_element)


def standardize(array):
    mean = np.mean(array)
    centered_array = array - mean
    std = np.std(array)
    if np.isclose(std, 0):
        return np.zeros_like(array)
    return centered_array / std
