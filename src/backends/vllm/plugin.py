from __future__ import annotations

from ..plugin import BackendPlugin
from .analyzer import VllmAnalyzer
from .parser import LATENCY_METRIC
from .runner import DEFAULTS, VllmRunner


def create_plugin() -> BackendPlugin:
    """Create the vLLM prefill/decode runner/analyzer plugin descriptor."""

    return BackendPlugin(
        name="vllm",
        runner_factory=VllmRunner,
        analyzer_factory=VllmAnalyzer,
        description="vLLM prefill/decode with KV handoff over NIXL (CPU serving simulator)",
        supports_network=True,
        profile_options=frozenset({*DEFAULTS, "timing_args", "kv_transfer_timing", "env"}),
        profile_mapping_options=frozenset({"timing_args", "kv_transfer_timing", "env"}),
        profile_dimensions=frozenset({"threads"}),
        graph_dimensions=frozenset({"threads", "prompt_length"}),
        primary_metric=LATENCY_METRIC,
    )
