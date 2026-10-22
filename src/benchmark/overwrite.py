from __future__ import annotations

from pathlib import Path
import shutil
from typing import Callable

from ..common.console import Colors
from .contracts import BenchmarkConfig


class OverwriteDeclined(Exception):
    """Raised when the user declines replacing an existing benchmark directory."""


def test_directory_path(config: BenchmarkConfig) -> Path:
    """Return the filesystem directory for a benchmark name."""

    return config.log_directory / config.output_name


def confirm_overwrite(
    config: BenchmarkConfig,
    *,
    input_fn: Callable[[str], str] = input,
    colors: Colors,
    force: bool = False,
) -> None:
    """Prompt before deleting an existing named benchmark directory; with
    ``force``, delete it without asking."""

    directory = test_directory_path(config)
    if not directory.exists():
        return
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"benchmark output path is not a directory: {directory}")

    if force:
        shutil.rmtree(directory)
        return
    answer = input_fn(
        f"{colors.yellow}Benchmark directory already exists: {directory}\n"
        f"Delete and overwrite it? [y/N] {colors.reset}"
    ).strip().lower()
    if answer not in {"y", "yes"}:
        raise OverwriteDeclined(f"benchmark overwrite declined: {directory}")
    shutil.rmtree(directory)
