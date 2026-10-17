from __future__ import annotations

from ..plugin import BackendPlugin
from .analyzer import NixlAnalyzer
from .runner import NixlRunner


def create_plugin() -> BackendPlugin:
    """Create the NIXL runner/analyzer plugin descriptor."""

    return BackendPlugin(
        name="nixl",
        runner_factory=NixlRunner,
        analyzer_factory=NixlAnalyzer,
        description="Buffer-transfer benchmark using NIXLBench (UCX, LIBFABRIC or Mooncake)",
        supports_network=True,
        profile_options=frozenset(
            {
                "iterations",
                "warmup_iterations",
                "run_timeout_seconds",
                "gpu",
                "backend",
                "parallelism",
                "args",
                "env",
            }
        ),
        profile_mapping_options=frozenset({"args", "env"}),
        profile_dimensions=frozenset({"threads", "block_size", "batch_size"}),
        graph_dimensions=frozenset({"threads", "block_size", "batch_size"}),
        renamed_profile_options={
            "warmup_runs": "execution.nixl.warmup_iterations",
            "repetitions": "execution.runs",
        },
    )
