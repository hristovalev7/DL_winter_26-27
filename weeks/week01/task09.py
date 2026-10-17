import numpy as np


def main():
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    matrix_scalar_result = matrix + 10
    print(matrix_scalar_result)

    row_vector = np.array([10, 20, 30])
    matrix_row_vector_result = matrix + row_vector
    print(matrix_row_vector_result)

    column_vector = np.array([[100], [200]])
    matrix_column_vector_result = matrix + column_vector
    print(matrix_column_vector_result)

    incompatible = np.array([1, 2, 3, 4])
    print(matrix + incompatible)
