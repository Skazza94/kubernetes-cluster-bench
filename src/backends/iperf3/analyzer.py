from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

from ...graphing.contracts import (
    AnalysisRequest,
    GraphPoint,
    Observation,
    ResultAnalyzer,
)
from ...graphing.selection import numeric_dimension, require_dimensions, select_observations
from ...graphing.statistics import mean, percentile, standard_deviation
from .parser import Iperf3Parser


class Iperf3Analyzer(ResultAnalyzer):
    """Aggregate iperf3 interval and repeated-run statistics."""

    def __init__(self, parser: Iperf3Parser | None = None) -> None:
        self.parser = parser or Iperf3Parser()

    def discover(self, raw_directory: Path) -> tuple[Path, ...]:
        return self.parser.discover(raw_directory)

    def parse(self, run_directory: Path) -> tuple[Observation, ...]:
        return self.parser.parse(run_directory)

    def aggregate(
        self,
        observations: tuple[Observation, ...],
        request: AnalysisRequest,
    ) -> tuple[GraphPoint, ...]:
        selected = select_observations(observations, request)
        require_dimensions(selected, request.series_dimensions)
        valid = tuple(
            observation
            for observation in selected
            if observation.metadata.get("valid") is True
        )
        if not valid:
            return ()

        run_groups: dict[tuple[str, float, tuple[str, ...]], list[Observation]] = defaultdict(list)
        for observation in valid:
            x_value = numeric_dimension(observation, request.x_dimension)
            series_key = _series_key(observation, request.series_dimensions)
            run_groups[(observation.run_id, x_value, series_key)].append(observation)

        run_points: dict[tuple[float, tuple[str, ...]], list[tuple[float, float, str]]] = defaultdict(list)
        for (run_id, x_value, series_key), group in run_groups.items():
            values = [observation.value for observation in group]
            y_value = (
                percentile(values, request.percentile)
                if request.percentile is not None
                else mean(values)
            ) / 1e9
            interval_error = standard_deviation(values) / 1e9
            first = group[0]
            x_label = _x_label(first, request.x_dimension, x_value)
            run_points[(x_value, series_key)].append(
                (y_value, interval_error, x_label)
            )

        points: list[GraphPoint] = []
        for (x_value, series_key), values in run_points.items():
            y_values = [value[0] for value in values]
            y_value = mean(y_values)
            error = (
                standard_deviation(y_values)
                if len(y_values) > 1
                else values[0][1]
            )
            points.append(
                GraphPoint(
                    x_value=x_value,
                    y_value=y_value,
                    x_label=values[0][2],
                    series_key=series_key,
                    y_error=(error, error),
                )
            )
        points.sort(key=lambda point: (point.x_value, point.series_key))
        return tuple(points)


def _series_key(
    observation: Observation,
    series_dimensions: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        f"{dimension}={observation.dimensions[dimension]}"
        for dimension in series_dimensions
    )


def _x_label(observation: Observation, dimension: str, x_value: float) -> str:
    labels = observation.metadata.get("dimension_labels")
    if isinstance(labels, dict):
        label = labels.get(dimension)
        if isinstance(label, str):
            return label
    return str(int(x_value)) if x_value.is_integer() else str(x_value)
