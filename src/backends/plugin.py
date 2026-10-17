from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from ..benchmark.contracts import BenchmarkRunner
from ..graphing.contracts import ResultAnalyzer


class BackendCapabilityError(RuntimeError):
    """Raised when a plugin does not provide a requested capability."""


@dataclass(frozen=True)
class BackendPlugin:
    """One backend entry shared by the benchmark and graphing frontends."""

    name: str
    runner_factory: Callable[[], BenchmarkRunner] | None = None
    analyzer_factory: Callable[[], ResultAnalyzer] | None = None
    description: str = ""
    profile_options: frozenset[str] = frozenset()
    profile_mapping_options: frozenset[str] = frozenset()
    profile_dimensions: frozenset[str] = frozenset()
    graph_dimensions: frozenset[str] = frozenset()
    # Options that were renamed, mapped to their new location, so old profiles
    # get a pointer instead of a generic unknown-field error.
    renamed_profile_options: Mapping[str, str] = field(default_factory=dict)
    supports_network: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("backend plugin name must not be empty")

    def create_runner(self) -> BenchmarkRunner:
        if self.runner_factory is None:
            raise BackendCapabilityError(
                f"backend {self.name!r} does not support benchmark execution"
            )
        return self.runner_factory()

    def create_analyzer(self) -> ResultAnalyzer:
        if self.analyzer_factory is None:
            raise BackendCapabilityError(
                f"backend {self.name!r} does not support result analysis"
            )
        return self.analyzer_factory()
