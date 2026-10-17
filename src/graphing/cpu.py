from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Sequence

from .contracts import Observation


CPU_METRIC = "cpu"
PER_GB_METRIC = "cpu-per-gb"
CPU_METRICS = (CPU_METRIC, PER_GB_METRIC)
SIDE_DIMENSION = "side"
SIDES = ("initiator", "target")

Y_LABELS = {
    CPU_METRIC: "CPU (cores)",
    PER_GB_METRIC: "CPU per GB moved (core-seconds)",
}
TITLES = {
    CPU_METRIC: "CPU",
    PER_GB_METRIC: "CPU per GB",
}


def cpu_observations(observations: Sequence[Observation], metric: str) -> tuple[Observation, ...]:
    """Turn throughput observations into per-side CPU observations.

    Each throughput observation that carries CPU samples (``cpu_cores_<side>``
    in its metadata) becomes one observation per side with a ``side``
    dimension, so the backend's own aggregation (labels, repetitions, error
    bars) applies unchanged. The analyzers divide values by 1e9, so values are
    stored times 1e9: cores for ``cpu``, and for ``cpu-per-gb`` the cores over
    the throughput in GB/s, which is the CPU time spent per GB moved.

    The throughput of ``cpu-per-gb`` is the mean of all valid observations
    that share the CPU window (a run's iperf3 intervals, for example), so the
    cost is not inflated by averaging the inverse of noisy intervals.
    """

    if metric not in CPU_METRICS:
        raise ValueError(f"unknown CPU metric {metric!r}")
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for observation in observations:
        if observation.metric == "throughput" and observation.metadata.get("valid") is not False:
            groups[_window_key(observation)].append(observation.value)
    derived: list[Observation] = []
    for observation in observations:
        if observation.metric != "throughput":
            continue
        for side in SIDES:
            cores = observation.metadata.get(f"cpu_cores_{side}")
            if isinstance(cores, bool) or not isinstance(cores, (int, float)):
                continue
            if metric == CPU_METRIC:
                value = float(cores)
            else:
                rates = groups.get(_window_key(observation))
                if not rates or sum(rates) <= 0:
                    continue
                gigabytes_per_second = sum(rates) / len(rates) / 8e9
                value = float(cores) / gigabytes_per_second
            derived.append(
                replace(
                    observation,
                    value=value * 1e9,
                    dimensions={**observation.dimensions, SIDE_DIMENSION: side},
                )
            )
    if not derived:
        raise ValueError(
            "no CPU samples in this dataset; runs record them from this version on, "
            "so run the benchmark again"
        )
    return tuple(derived)


def _window_key(observation: Observation) -> tuple[object, ...]:
    """Observations of one run and point share one CPU window."""

    return (
        observation.run_id,
        tuple(sorted(observation.dimensions.items())),
        observation.metadata.get("cpu_window_s"),
    )
