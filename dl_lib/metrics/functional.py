import numpy as np


def apply_threshold(probabilities, threshold):
    return probabilities > threshold


def confusion_matrix(true_labels, predicted_labels):
    true_positives = np.sum((true_labels == 1) & (predicted_labels == 1))
    true_negatives = np.sum((true_labels == 0) & (predicted_labels == 0))
    false_positives = np.sum((true_labels == 0) & (predicted_labels == 1))
    false_negatives = np.sum((true_labels == 1) & (predicted_labels == 0))
    return {
        "tp": true_positives,
        "tn": true_negatives,
        "fp": false_positives,
        "fn": false_negatives,
    }


def accuracy(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    return (cm["tp"] + cm["tn"]) / (cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"])


def precision(true_labels, precited_labels):
    cm = confusion_matrix(true_labels, precited_labels)
    if cm["tp"] + cm["fp"] == 0:
        return 0
    return cm["tp"] / (cm["tp"] + cm["fp"])


def recall(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    if cm["tp"] + cm["fn"] == 0:
        return 0
    return cm["tp"] / (cm["tp"] + cm["fn"])


def f1_score(true_labels, predicted_labels):
    prec = precision(true_labels, predicted_labels)
    rec = recall(true_labels, predicted_labels)
    if prec == 0 and rec == 0:
        return 0
    return 2 * prec * rec / (prec + rec)
