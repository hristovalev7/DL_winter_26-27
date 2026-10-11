import sys
import unittest

from weeks import main


class TestMain(unittest.TestCase):
    def test_when_main_called_then_always_returns_zero(self):
        expected = 0
        actual = main.main()
        self.assertEqual(actual, expected)


class TestParseCommandLineArguments(unittest.TestCase):
    def test_when_no_args_then_returns_one_one(self):
        sys.argv = ["run.py"]
        expected = (1, 1)

        actual = main.parse_command_line_arguments()

        self.assertEqual(actual, expected)

    def test_when_only_weeks_given_then_returns_the_week_and_one(self):
        expected_week = 5
        sys.argv = ["run.py", "-w", str(expected_week)]
        expected = (expected_week, 1)

        actual = main.parse_command_line_arguments()

        self.assertEqual(actual, expected)
