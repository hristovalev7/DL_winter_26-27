import numpy as np

from dl_lib.losses.functional import binary_cross_entropy, categorical_cross_entropy


def main():
    y_true = np.array([1., 0., 1.])
    predicted_labels = np.array([0.9, 0.2, 0.6])
    print(f"Binary cross-entropy on y_true={y_true}, p={predicted_labels}: {binary_cross_entropy(y_true, predicted_labels)}")
    y_true = np.array([1.0])
    confidently_right = np.array([0.999])
    confidently_wrong = np.array([0.001])
    print(f"y_true={y_true[0]}, p={confidently_right[0]} (confidently right): loss={binary_cross_entropy(y_true, confidently_right)}")
    print(f"y_true={y_true[0]}, p={confidently_wrong[0]} (confidently right): loss={binary_cross_entropy(y_true, confidently_wrong)}")
    print()
    true_labels = np.array([0, 1, 2])
    predicted_probabilities = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6]
    ])
    print(f"Multiclass cross-entropy on y_true={true_labels}: {categorical_cross_entropy(true_labels, predicted_probabilities)}")
    y_true = np.array([0])
    confidently_right = np.array([[0.999]])
    confidently_wrong = np.array([[0.001]])
    print(f"y_true={y_true[0]}, p={confidently_right[0]} (confidently right): loss={categorical_cross_entropy(y_true, confidently_right)}")
    print(f"y_true={y_true[0]}, p={confidently_wrong[0]} (confidently wrong): loss={categorical_cross_entropy(y_true, confidently_wrong)}")
