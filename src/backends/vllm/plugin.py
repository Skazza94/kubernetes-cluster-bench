from __future__ import annotations

from ..plugin import BackendPlugin
from .analyzer import VllmAnalyzer
from .parser import LATENCY_METRIC
from .runner import DEFAULTS, VllmRunner
from .config import MAPPING_OPTIONS


def create_plugin() -> BackendPlugin:
    """Create the vLLM prefill/decode runner/analyzer plugin descriptor."""

    return BackendPlugin(
        name="vllm",
        runner_factory=VllmRunner,
        analyzer_factory=VllmAnalyzer,
        description="vLLM simulated compute with explicit native KV transport",
        supports_network=True,
        profile_options=frozenset({*DEFAULTS, *MAPPING_OPTIONS, "timing_args"}),
        profile_mapping_options=MAPPING_OPTIONS,
        profile_json_mapping_options=frozenset({"timing_args"}),
        profile_dimensions=frozenset({"threads"}),
        graph_dimensions=frozenset({"threads", "prompt_length", "side"}),
        primary_metric=LATENCY_METRIC,
    )
