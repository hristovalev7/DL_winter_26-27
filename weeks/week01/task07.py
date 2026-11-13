import numpy


def main():
    matrix = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(matrix[1, :])
    print(matrix[:, 2])
    print(matrix[1:3, 0:3])
    print(matrix > 6)
    print(matrix[matrix > 6])
