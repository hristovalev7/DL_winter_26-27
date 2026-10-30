import numpy as np

from dl_lib.metrics.functional import accuracy, precision, recall, f1_score


def main():
    rng = np.random.default_rng()

    y_true = rng.integers(0, 2, size=10)
    y_pred = y_true.copy()
    ones = np.flatnonzero(y_pred)
    chosen_to_flip = rng.choice(ones, size=2, replace=False)
    y_pred[chosen_to_flip] = 0

    y_true_big = rng.permutation(np.array(990 * [0] + 10 * [1]))
    y_pred_big = y_true_big.copy()
    ones_big = np.flatnonzero(y_pred_big)
    chosen_to_flip = rng.choice(ones_big, size=3, replace=False)
    y_pred_big[chosen_to_flip] = 0

    print("Balanced dataset:")
    print(f"accuracy: {accuracy(y_true, y_pred)}")
    print(f"precision: {precision(y_true, y_pred)}")
    print(f"recall: {recall(y_true, y_pred)}")
    print(f"f1: {f1_score(y_true, y_pred)}")

    print("Imbalanced dataset:")
    print(f"accuracy: {accuracy(y_true_big, y_pred_big)}")
    print(f"precision: {precision(y_true_big, y_pred_big)}")
    print(f"recall: {recall(y_true_big, y_pred_big)}")
    print(f"f1: {f1_score(y_true_big, y_pred_big)}")
