from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys
from typing import NoReturn, Sequence

from ..backends.registry import BackendRegistry, default_registry
from ..common.console import format_error
from .contracts import AnalysisRequest, DimensionValue
from .cpu import CPU_METRICS, SIDE_DIMENSION, TITLES, Y_LABELS, cpu_observations
from .formatting import GraphArgumentParser, GraphHelpFormatter
from .loader import load_observations
from .renderer import GraphRenderConfig, render_graph


THROUGHPUT_METRIC = "throughput"


def build_parser() -> GraphArgumentParser:
    parser = GraphArgumentParser(
        prog="plot-benchmark.sh",
        usage="%(prog)s --backend BACKEND --name NAME --x DIMENSION [options]",
        description="Plot benchmark throughput from backend raw artifacts.",
        formatter_class=GraphHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--backend", required=True, metavar="BACKEND")
    parser.add_argument(
        "--name",
        required=True,
        metavar="NAME",
        help="Benchmark log directory name.",
    )
    parser.add_argument("-d", "--dir", dest="directory", type=Path, default=Path("logs"))
    parser.add_argument("--x", required=True, dest="x_dimension", metavar="DIMENSION")
    parser.add_argument(
        "--x-log",
        type=positive_float,
        default=None,
        metavar="BASE",
        help="Use logarithmic X-axis scaling with the specified base.",
    )
    parser.add_argument(
        "--y",
        default=THROUGHPUT_METRIC,
        choices=(THROUGHPUT_METRIC, *CPU_METRICS),
        metavar="METRIC",
        help=(
            "Metric to plot: throughput, cpu (cores used by each side) or "
            "cpu-per-gb (CPU time each side spends per GB moved, in core-seconds). The CPU "
            "metrics plot one line per side; --filter side=initiator or "
            "side=target keeps one."
        ),
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Fix a dimension to one value; use for dimensions that are not "
            "the X axis or a series. May be repeated."
        ),
    )
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        dest="series_dimensions",
        metavar="DIMENSION",
        help=(
            "Plot a separate line for each value of this dimension; use instead "
            "of filtering when comparison is desired. May be repeated."
        ),
    )
    parser.add_argument("--percentile", type=percentile, default=None, metavar="PERCENT")
    parser.add_argument("--max-throughput", type=positive_float, default=None, metavar="GBIT/S")
    parser.add_argument("--output", type=Path, default=None, metavar="FILE")
    parser.add_argument("--title", default=None, metavar="TEXT")
    return parser


def main(
    arguments: Sequence[str] | None = None,
    registry: BackendRegistry | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(arguments)
    selected_registry = registry or default_registry()
    try:
        plugin = selected_registry.get(args.backend)
        cpu_metric = args.y in CPU_METRICS
        if cpu_metric and args.max_throughput is not None:
            raise ValueError("--max-throughput only applies to --y throughput")
        _validate_dimensions(
            plugin.graph_dimensions | ({SIDE_DIMENSION} if cpu_metric else set()),
            args.x_dimension,
            args.filter,
            args.series_dimensions,
        )
        x_dimension = _canonical_dimension(args.x_dimension)
        filters = _canonical_filters(args.filter)
        series_dimensions = tuple(
            _canonical_dimension(dimension) for dimension in args.series_dimensions
        )
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.name) is None:
            raise ValueError(
                "name must contain only letters, numbers, dots, underscores, and hyphens"
            )
        if cpu_metric and SIDE_DIMENSION not in filters_by_name(filters):
            series_dimensions = (*series_dimensions, SIDE_DIMENSION)
        request = AnalysisRequest(
            metric=THROUGHPUT_METRIC,
            x_dimension=x_dimension,
            filters=_parse_filters(filters),
            series_dimensions=series_dimensions,
            percentile=args.percentile,
        )
        test_directory = args.directory / args.name
        observations = load_observations(
            args.backend,
            test_directory,
            selected_registry,
        )
        if cpu_metric:
            observations = cpu_observations(observations, args.y)
        analyzer = plugin.create_analyzer()
        points = analyzer.aggregate(observations, request)
        if not points:
            raise ValueError("no finite observations contain the selected options")
        title = args.title or _graph_title(
            observations,
            request,
            args.x_dimension,
            TITLES.get(args.y, "Throughput"),
        )
        output_path = args.output or (
            test_directory / "graphs" / default_graph_filename(args)
        )
        render_graph(
            points,
            GraphRenderConfig(
                output_path=output_path,
                x_label=_label(args.x_dimension),
                y_label=Y_LABELS.get(args.y, "Throughput (Gbit/s)"),
                title=title,
                reference_y=args.max_throughput,
                reference_label=(
                    f"{args.max_throughput:g} Gbit/s max"
                    if args.max_throughput is not None
                    else None
                ),
                x_log_base=args.x_log,
            ),
        )
        print(f"Graph written to {output_path}")
    except (OSError, RuntimeError, ValueError) as error:
        print(format_error(str(error), sys.stderr), file=sys.stderr)
        return 2
    return 0


MAX_FILENAME_LENGTH = 200


def default_graph_filename(args: argparse.Namespace) -> str:
    # By default, generate a name that keeps the sweep parameters inside.
    parts = [f"{args.y}-{_slug(args.x_dimension)}"]
    filters = sorted(_split_filter(raw) for raw in args.filter)
    parts.extend(f"{_slug(name)}-{_slug(value)}" for name, value in filters)
    if args.series_dimensions:
        parts.append("by-" + "-".join(_slug(name) for name in args.series_dimensions))
    if args.percentile is not None:
        parts.append(f"p{args.percentile:g}")
    if args.x_log is not None:
        parts.append(f"xlog{args.x_log:g}")
    if args.max_throughput is not None:
        parts.append(f"max{args.max_throughput:g}")
    name = "__".join(parts)
    if len(name) > MAX_FILENAME_LENGTH:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[: MAX_FILENAME_LENGTH - 9]}-{digest}"
    return f"{name}.png"


def filters_by_name(values: Sequence[str]) -> set[str]:
    return {value.split("=", 1)[0] for value in values}


def _split_filter(raw: str) -> tuple[str, str]:
    name, _, value = raw.partition("=")
    return name.strip(), value.strip()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._]+", "-", text).strip("-") or "x"


def percentile(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not 0 <= parsed <= 100:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _canonical_filters(values: Sequence[str]) -> tuple[str, ...]:
    canonical: list[str] = []
    for value in values:
        if "=" not in value:
            canonical.append(value)
            continue
        name, raw_value = value.split("=", 1)
        canonical.append(f"{_canonical_dimension(name)}={raw_value}")
    return tuple(canonical)


def _graph_title(
    observations: tuple,
    request: AnalysisRequest,
    x_dimension: str,
    metric_title: str = "Throughput",
) -> str:
    fixed: dict[str, DimensionValue] = dict(request.filters)
    dimensions = {
        name
        for observation in observations
        for name in observation.dimensions
    }
    for dimension in dimensions:
        if dimension == request.x_dimension or dimension in request.series_dimensions:
            continue
        values = {
            observation.dimensions[dimension]
            for observation in observations
            if dimension in observation.dimensions
        }
        if len(values) == 1:
            fixed.setdefault(dimension, next(iter(values)))
    qualifiers = ", ".join(
        f"{_title_dimension(name)}={_format_title_value(name, value)}"
        for name, value in sorted(fixed.items())
    )
    title = f"{metric_title} by {_label(x_dimension)}"
    return f"{title} ({qualifiers})" if qualifiers else title


def _title_dimension(dimension: str) -> str:
    return _canonical_dimension(dimension).replace("_", " ")


def _format_title_value(dimension: str, value: DimensionValue) -> str:
    if dimension in {"block_size", "block_size_bytes"}:
        number = int(value)
        for suffix, multiplier in (("G", 1024**3), ("M", 1024**2), ("K", 1024)):
            if number % multiplier == 0:
                return f"{number // multiplier}{suffix}"
    return str(value)


def _validate_dimensions(
    supported: frozenset[str],
    x_dimension: str,
    filter_values: Sequence[str],
    series_dimensions: Sequence[str],
) -> None:
    requested = {x_dimension, *series_dimensions}
    for value in filter_values:
        if "=" not in value:
            raise ValueError("--filter must use NAME=VALUE")
        requested.add(value.split("=", 1)[0])
    unsupported = sorted(requested - supported)
    if unsupported:
        names = ", ".join(unsupported)
        available = ", ".join(sorted(supported)) or "none"
        raise ValueError(
            f"unsupported graph dimension(s): {names}; "
            f"backend supports: {available}"
        )


def _parse_filters(values: Sequence[str]) -> dict[str, DimensionValue]:
    filters: dict[str, DimensionValue] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--filter must use NAME=VALUE")
        name, raw_value = value.split("=", 1)
        if not name or not raw_value:
            raise ValueError("--filter must have a non-empty name and value")
        filters[name] = _parse_scalar(raw_value, name)
    return filters


def _parse_scalar(value: str, dimension: str | None = None) -> DimensionValue:
    if dimension in {"block_size", "block_size_bytes"}:
        suffix = value[-1:].upper()
        multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}.get(suffix, 1)
        number = value[:-1] if suffix in {"K", "M", "G"} else value
        if suffix in {"K", "M", "G"} or number.isdigit():
            try:
                return int(number) * multiplier
            except ValueError:
                pass
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _canonical_dimension(dimension: str) -> str:
    if dimension == "block_size":
        return "block_size_bytes"
    return dimension


def _format_axis_label(dimension: str, value: float) -> str:
    if dimension in {"block_size", "block_size_bytes"}:
        number = int(value)
        for suffix, multiplier in (("G", 1024**3), ("M", 1024**2), ("K", 1024)):
            if number % multiplier == 0:
                return f"{number // multiplier}{suffix}"
    return str(int(value)) if value.is_integer() else str(value)


def _label(dimension: str) -> str:
    if dimension == "block_size_bytes":
        dimension = "block_size"
    return dimension.replace("_", " ").capitalize()
