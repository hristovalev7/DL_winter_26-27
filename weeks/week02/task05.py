import numpy as np

from dl_lib.metrics.functional import apply_threshold


def main():
    probabilities = np.array([0.1, 0.8, 0.4, 0.7, 0.9])
    for t in probabilities:
        resulting_predictions = apply_threshold(probabilities, t)
        print(f"Threshold={t}: predictions=[", end="")
        print(*resulting_predictions, sep=", ", end="")
        print("]")
