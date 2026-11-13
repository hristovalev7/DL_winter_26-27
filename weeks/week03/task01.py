from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import initialize_weights


def main():
    example_dataset = create_dataset(4)
    draw1 = initialize_weights(0, 100, 1)
    draw2 = initialize_weights(0, 10, 1)
    print(f"Dataset: {example_dataset}")
    print(f"initialize_weights(0, 100, 1): {draw1}")
    print(f"initialize_weights(0, 10, 1): {draw2}")
