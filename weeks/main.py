import argparse


def main() -> int:
    return 0


def parse_command_line_arguments() -> tuple[int, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--week", type=int, default=1)
    args = parser.parse_args()
    week = args.week
    return (week, 1)
