import numpy as np


def initialize_weights(low, high, size, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return rng.uniform(low, high, size)
