import numpy as np
from matplotlib import pyplot as plt


def main():
    rng = np.random.default_rng(7)
    uniform_distribution = rng.uniform(1, 100, 10000)
    normal_distribution = rng.normal(0, 1, 10000)
    exponential_distribution = rng.exponential(1, 1000)
    print("Uniform distribution")
    print(
        f"mean={np.mean(uniform_distribution)}, std={np.std(uniform_distribution)}, min={np.min(uniform_distribution)}, max={np.max(uniform_distribution)}"
    )
    print()
    print("Normal distribution")
    print(
        f"mean={np.mean(normal_distribution)}, std={np.std(normal_distribution)}, min={np.min(normal_distribution)}, max={np.max(normal_distribution)}"
    )
    print()
    print("Exponential distribution")
    print(
        f"mean={np.mean(exponential_distribution)}, std={np.std(exponential_distribution)}, min={np.min(exponential_distribution)}, max={np.max(exponential_distribution)}"
    )
    (_figure, (ax1, ax2, ax3)) = plt.subplots(1, 3)
    ax1.set_title("Uniform")
    ax1.set_ylabel("count")
    ax1.set_xlabel("value")
    ax1.hist(uniform_distribution, 33)
    ax2.set_title("Normal")
    ax2.set_ylabel("count")
    ax2.set_xlabel("value")
    ax2.hist(normal_distribution, 33)
    ax3.set_title("Exponential")
    ax3.set_ylabel("count")
    ax3.set_xlabel("value")
    ax3.hist(exponential_distribution, 33)
    plt.show()
