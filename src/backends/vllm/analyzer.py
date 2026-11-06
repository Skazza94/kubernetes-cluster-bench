from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ...graphing.contracts import AnalysisRequest, GraphPoint, Observation, ResultAnalyzer
from ...graphing.selection import numeric_dimension, require_dimensions, select_observations
from ...graphing.statistics import mean, standard_deviation
from .parser import VllmParser


class VllmAnalyzer(ResultAnalyzer):
    """Aggregate request latencies in milliseconds and run metrics in their units."""

    def __init__(self, parser: VllmParser | None = None) -> None:
        self.parser = parser or VllmParser()

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
        hidden = sorted(
            name
            for name in {name for observation in selected for name in observation.dimensions}
            if name != request.x_dimension
            and name not in request.series_dimensions
            and len({observation.dimensions.get(name) for observation in selected}) > 1
        )
        if hidden:
            raise ValueError(f"vLLM dimensions vary ({', '.join(hidden)}); add --filter or --series")
        if len({observation.unit for observation in selected}) > 1:
            raise ValueError("vLLM graph mixes measurement units")
        groups: dict[tuple[float, tuple[str, ...]], list[float]] = defaultdict(list)
        for observation in selected:
            x_value = numeric_dimension(observation, request.x_dimension)
            series = tuple(
                f"{name}={observation.dimensions[name]}" for name in request.series_dimensions
            )
            scale = 1.0 if observation.sample_kind == "run" else 1000.0 if observation.unit == "s" else 1e-9
            groups[(x_value, series)].append(observation.value * scale)
        points = []
        for (x_value, series), values in groups.items():
            error = standard_deviation(values) if len(values) > 1 else 0.0
            points.append(GraphPoint(
                x_value=x_value,
                y_value=mean(values),
                x_label=str(int(x_value)) if x_value.is_integer() else str(x_value),
                series_key=series,
                y_error=(error, error),
            ))
        points.sort(key=lambda point: (point.x_value, point.series_key))
        return tuple(points)
