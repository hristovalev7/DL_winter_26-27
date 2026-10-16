import numpy as np


def main():
    matrix_a = np.array([[1, 2], [3, 4]])
    matrix_b = np.array([[5, 6], [7, 8]])
    print("A:")
    print(matrix_a)
    print("B:")
    print(matrix_b)

    element_wise_product = matrix_a * matrix_b
    print("Elementwise product:")
    print(element_wise_product)
    matrix_multiplication_product = matrix_a @ matrix_b
    print("Matrix multiplication product:")
    print(matrix_multiplication_product)

    matrix_a_transpose = matrix_a.T
    print("Transpose of matrix A")
    print(matrix_a_transpose)
