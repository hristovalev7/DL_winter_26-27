import argparse
import importlib


def main() -> int:
    (week, task) = parse_command_line_arguments()
    task_lib = importlib.import_module(f"weeks.week{week:02d}.task{task:02d}")
    return task_lib.main()


def parse_command_line_arguments() -> tuple[int, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--week", type=int, default=1)
    parser.add_argument("-t", "--task", type=int, default=1)
    args = parser.parse_args()
    week = args.week
    task = args.task
    return (week, task)
