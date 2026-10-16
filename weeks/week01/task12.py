import numpy as np
from matplotlib import pyplot as plt


def main():
    num_of_points = 1000
    x_axis = np.linspace(0, 2 * np.pi, num_of_points)
    sin_y_axis = np.sin(x_axis)
    cos_y_axis = np.cos(x_axis)
    (figure, axes) = plt.subplots(1, 2)
    axes[0].plot(x_axis, sin_y_axis)
    axes[0].plot(x_axis, cos_y_axis)

    rng = np.random.default_rng(77)
    samples = rng.normal(np.mean(x_axis), 2, num_of_points)
    axes[1].hist(samples, bins=30)
    plt.show()
