import numpy as np


def apply_threshold(probabilities, threshold):
    return probabilities > threshold


def confusion_matrix(true_labels, predicted_labels):
    true_positives = np.sum((true_labels == 1) & (predicted_labels == 1))
    true_negatives = np.sum((true_labels == 0) & (predicted_labels == 0))
    false_positives = np.sum((true_labels == 0) & (predicted_labels == 1))
    false_negatives = np.sum((true_labels == 1) & (predicted_labels == 0))
    return {"tp": true_positives, "tn": true_negatives, "fp": false_positives, "fn": false_negatives}
