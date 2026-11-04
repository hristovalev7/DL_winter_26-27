import numpy as np

from dl_lib.activations.functional import naive_softmax, softmax


def main():
    small_scores = np.array([1.0, 2.0, 3.0])
    softmax_result = softmax(small_scores)
    sum_of_probabilities = np.sum(softmax_result)
    print(f"scores {small_scores}")
    print(f"softmax {softmax_result}")
    print(f"sum of probabilities {sum_of_probabilities}")
    print()
    large_scores = np.array([1000.0, 1001.0, 1002.0])
    naive_softmax_result = naive_softmax(large_scores)
    stable_softmax_result = softmax(large_scores)
    print(f"large scores: {large_scores}")
    print(f"naive softmax: {naive_softmax_result}")
    print(f"stable softmax: {stable_softmax_result}")
