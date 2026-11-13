import numpy as np

from dl_lib.metrics.functional import confusion_matrix


def main():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    cm = confusion_matrix(y_true, y_pred)
    print(f"True positives: {cm['tp']}")
    print(f"True negatives: {cm['tn']}")
    print(f"False positives: {cm['fp']}")
    print(f"False negatives: {cm['fn']}")
