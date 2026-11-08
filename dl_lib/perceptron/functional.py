import numpy as np


def initialize_weights(low, high, size, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size)
