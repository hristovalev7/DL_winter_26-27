import numpy as np


def create_dataset(n):
    x = np.arange(n)
    dataset = []
    for i in range(n):
        dataset.append((i, i * 2))
    return dataset

