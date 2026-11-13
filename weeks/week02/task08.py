import numpy as np

from dl_lib.metrics.functional import accuracy, f1_score, precision, recall


def main():
    rng = np.random.default_rng()
    y_true = np.array(990 * [0] + 10 * [1])
    rng.shuffle(y_true)
    y_pred = np.array(1000 * [0])
    print(f"accuracy: {accuracy(y_true, y_pred)}")
    print(f"precision: {precision(y_true, y_pred)}")
    print(f"recall: {recall(y_true, y_pred)}")
    print(f"f1: {f1_score(y_true, y_pred)}")
