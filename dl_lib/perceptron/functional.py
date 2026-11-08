import numpy as np


def initialize_weights(low, high, size, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size)


def forward(x, weights, bias=0.0, activation=None):
    if activation is not None:
        return activation(x @ weights + bias)
    return x @ weights + bias
