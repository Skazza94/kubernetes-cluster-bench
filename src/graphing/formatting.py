from __future__ import annotations

import argparse
import sys
from typing import NoReturn

from ..common.console import format_error


class GraphHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=32, width=100)


class GraphArgumentParser(argparse.ArgumentParser):
    """Argument parser with formatted graphing errors."""

    def error(self, message: str) -> NoReturn:
        print(format_error(message, sys.stderr), file=sys.stderr)
        print(file=sys.stderr)
        self.print_help(sys.stderr)
        self.exit(2)
