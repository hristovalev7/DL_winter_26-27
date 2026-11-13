import numpy


def main():
    one_dimensional = numpy.array([1, 2, 3, 4, 5, 6, 7])
    two_dimensional = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    three_dimensional = numpy.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
        ]
    )
    print(
        f"1D shape={one_dimensional.shape}, dtype={one_dimensional.dtype}, ndim={one_dimensional.ndim}, size={one_dimensional.size}"
    )
    print(one_dimensional)
    print(
        f"2D shape={two_dimensional.shape}, dtype={two_dimensional.dtype}, ndim={two_dimensional.ndim}, size={two_dimensional.size}"
    )
    print(two_dimensional)
    print(
        f"3D shape={three_dimensional.shape}, dtype={three_dimensional.dtype}, ndim={three_dimensional.ndim}, size={three_dimensional.size}"
    )
    print(three_dimensional)
