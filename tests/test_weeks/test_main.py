import sys
import unittest

import pytest

from weeks import main


class TestMain(unittest.TestCase):
    def setUp(self) -> None:
        self.original_argv_value = sys.argv

    def tearDown(self) -> None:
        sys.argv = self.original_argv_value

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_main_called_then_always_returns_zero(self):
        sys.argv = ["run.py"]
        expected = 0
        actual = main.main()
        self.assertEqual(actual, expected)

    def test_when_no_arguments_given_then_task_one_from_week_one_is_run(self):
        sys.argv = ["run.py"]
        expected = "Running task 1 from week 1."
        main.main()
        actual = self.capsys.readouterr().out
        self.assertIn(expected, actual)

    def test_when_invalid_week_or_task_given_then_throws_value_error(self):
        week = 100
        task = 1
        sys.argv = ["run.py", "-w", str(week), "-t", str(task)]
        with pytest.raises(ValueError) as e:
            main.main()
        actual_text = str(e.value)
        expected_text = f"Task {task:02d} does not exist in week {week:02d}!"
        self.assertIn(expected_text, actual_text)


class TestParseCommandLineArguments(unittest.TestCase):
    def setUp(self) -> None:
        self.original_argv_value = sys.argv

    def tearDown(self) -> None:
        sys.argv = self.original_argv_value

    def test_when_no_args_then_returns_one_one(self):
        sys.argv = ["run.py"]
        expected = (1, 1)

        actual = main.parse_command_line_arguments()

        self.assertEqual(actual, expected)

    def test_when_only_week_given_then_returns_the_week_and_one(self):
        expected_week = 5
        expected_task = 1
        sys.argv = ["run.py", "-w", str(expected_week)]
        expected = (expected_week, expected_task)

        actual = main.parse_command_line_arguments()

        self.assertEqual(actual, expected)

    def test_when_only_task_given_then_returns_one_and_the_task(self):
        expected_week = 1
        expected_task = 5
        sys.argv = ["run.py", "-t", str(expected_task)]
        expected = (expected_week, expected_task)

        actual = main.parse_command_line_arguments()

        self.assertEqual(actual, expected)

    def test_when_week_and_task_given_then_return_the_week_and_the_task(self):
        expected_week = 7
        expected_task = 5
        sys.argv = [
            "run.py",
            "--week",
            str(expected_week),
            "--task",
            str(expected_task),
        ]
        expected = (expected_week, expected_task)

        actual = main.parse_command_line_arguments()

        self.assertEqual(actual, expected)
