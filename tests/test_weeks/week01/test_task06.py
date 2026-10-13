import unittest

import numpy
import pytest

from weeks.week01 import task06


class TestTask06(unittest.TestCase):
    one_dimensional = numpy.array([1, 2, 3, 4, 5, 6, 7])
    two_dimensional = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    three_dimensional = numpy.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
        ]
    )

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_main_called_one_dimensional_array_printed(self):
        expected = f"{self.one_dimensional}"
        task06.main()
        actual = self.capsys.readouterr().out
        self.assertIn(expected, actual)

    def test_when_main_called_one_dimensional_array_properties_printed(self):
        expected = f"1D shape={self.one_dimensional.shape}, dtype={self.one_dimensional.dtype}, ndim={self.one_dimensional.ndim}, size={self.one_dimensional.size}"
        task06.main()
        actual = self.capsys.readouterr().out
        self.assertIn(expected, actual)

    def test_when_main_called_two_dimensional_array_printed(self):
        expected = f"{self.two_dimensional}"
        task06.main()
        actual = self.capsys.readouterr().out
        self.assertIn(expected, actual)

    def test_when_main_called_two_dimensional_array_properties_printed(self):
        expected = f"2D shape={self.two_dimensional.shape}, dtype={self.two_dimensional.dtype}, ndim={self.two_dimensional.ndim}, size={self.two_dimensional.size}"
        task06.main()
        actual = self.capsys.readouterr().out
        self.assertIn(expected, actual)

    def test_when_main_called_three_dimensional_array_printed(self):
        expected = f"{self.three_dimensional}"
        task06.main()
        actual = self.capsys.readouterr().out
        self.assertIn(expected, actual)

    def test_when_main_called_three_dimensional_array_properties_printed(self):
        expected = f"3D shape={self.three_dimensional.shape}, dtype={self.three_dimensional.dtype}, ndim={self.three_dimensional.ndim}, size={self.three_dimensional.size}"
        task06.main()
        actual = self.capsys.readouterr().out
        self.assertIn(expected, actual)
