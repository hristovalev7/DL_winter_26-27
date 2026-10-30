import numpy as np

from dl_lib.metrics.functional import precision, recall


def main():
    rng = np.random.default_rng()
    y_true = rng.integers(0, 2, size=1000)
    probabilities = rng.uniform(0., 1., size=1000)
    thresholds = np.linspace(0, 1, 11)
    for t in thresholds:
        y_pred = probabilities > t
        print(f"threshold={t:.1f}: precision={precision(y_true, y_pred):.4f}, recall={recall(y_true, y_pred):.4f}")
