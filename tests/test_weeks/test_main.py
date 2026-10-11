import unittest

from weeks import main


class TestMain(unittest.TestCase):
    def test_when_main_called_then_always_returns_zero(self):
        expected = 0
        actual = main.main()
        self.assertEqual(actual, expected)
