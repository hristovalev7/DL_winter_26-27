import matplotlib.pyplot as plt
import numpy as np


def main():
    points = 100
    x_vector = np.linspace(0, 10, points)
    y_vector = x_vector * 2 + 1

    rng = np.random.default_rng(seed=77)
    rng_vector = rng.normal(0, 1, points)

    y_vector = y_vector + rng_vector
    (figure, axis) = plt.subplots()
    axis.scatter(x_vector, y_vector)
    axis.set_title("Points")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    plt.show()