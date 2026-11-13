import matplotlib.pyplot as plt
import numpy as np

from dl_lib.metrics.functional import apply_threshold, precision, recall


def main():
    rng = np.random.default_rng()
    y_true = np.array([0] * 70 + [1] * 30)
    negative_scores = rng.uniform(0.0, 0.6, size=70)
    positive_scores = rng.uniform(0.4, 0.8, size=30)
    probabilities = np.concatenate([negative_scores, positive_scores])
    thresholds = np.linspace(0, 1, 20)
    precision_values = []
    recall_values = []
    for t in thresholds:
        y_pred = apply_threshold(probabilities, t)
        current_precision = precision(y_true, y_pred)
        current_recall = recall(y_true, y_pred)
        print(
            f"threshold={t:.2f}: precision={current_precision:.4f}, recall={current_recall:.4f}"
        )
        precision_values.append(current_precision)
        recall_values.append(current_recall)
    plt.plot(thresholds, precision_values, marker="o", label="precision")
    plt.plot(thresholds, recall_values, marker="o", label="recall")
    plt.xlabel("threshold")
    plt.ylabel("value")
    plt.title("Precision and recall as teh threshold rises")
    plt.grid(True)
    plt.legend()
    plt.show()
