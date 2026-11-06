from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Mapping, cast

from ...benchmark.cpu import cpu_metadata
from ...common.types import JsonValue
from ...graphing.contracts import Observation


JsonObject = Mapping[str, object]
_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)(?:\s+|$)"
)
_THREADS_RE = re.compile(r"Num threads \([^)]*\)\s*:\s*(\d+)")


class NixlParser:
    """Parse NIXLBench initiator tables into throughput observations."""

    def discover(self, raw_directory: Path) -> tuple[Path, ...]:
        directories = tuple(
            sorted(
                path
                for path in raw_directory.glob("t*")
                if path.is_dir() and any(path.glob("*-command.json"))
            )
        )
        if not directories:
            raise ValueError(f"no NIXL command metadata found in {raw_directory}")
        return directories

    def parse(self, run_directory: Path) -> tuple[Observation, ...]:
        command_files = sorted(run_directory.glob("*-command.json"))
        if not command_files:
            raise ValueError(f"no NIXL command metadata found in {run_directory}")
        observations: list[Observation] = []
        for command_path in command_files:
            metadata = _load_object(command_path)
            if "instance_files" in metadata:
                raise ValueError(
                    f"{command_path} was written by an earlier version that summed parallel "
                    "instances row by row, which overstates throughput; run it again"
                )
            if metadata.get("initiator_status", 0) != 0 or metadata.get("target_status", 0) != 0:
                raise ValueError(f"NIXL run failed according to {command_path}")
            if metadata.get("parallelism") == "processes":
                observations.extend(_process_observations(run_directory, metadata))
            else:
                observations.extend(_thread_observations(run_directory, metadata))
        if not observations:
            raise ValueError(f"no NIXL measurement rows found in {run_directory}")
        return tuple(observations)


def _thread_observations(run_directory: Path, metadata: JsonObject) -> list[Observation]:
    """Observations of a run that used one nixlbench pair: one per table row."""

    initiator_path = run_directory / _required_string(metadata, "initiator_file")
    target_path = run_directory / _required_string(metadata, "target_file")
    _require_files(initiator_path, target_path)
    text = initiator_path.read_text(encoding="utf-8", errors="replace")
    threads = _effective_threads(text, metadata)
    run_id = _run_id(metadata, threads)
    timing = _row_timing(metadata)
    observations = []
    for row_index, block_size, batch_size, gb_per_second in _table_rows(text):
        details: dict[str, JsonValue] = {
            "row_index": row_index,
            "initiator_file": initiator_path.name,
            "target_file": target_path.name,
        }
        timed_bytes, seen = timing.get((block_size, batch_size), (None, None))
        if timed_bytes is not None and seen is not None and gb_per_second > 0:
            # The row appears when its timed phase ends; the rate gives how long it ran.
            start = seen - timed_bytes / (gb_per_second * 1e9)
            details.update(cpu_metadata(run_directory, metadata, start, seen))
        observations.append(
            _observation(gb_per_second, threads, block_size, batch_size, run_id, details)
        )
    return observations


def _row_timing(metadata: JsonObject) -> dict[tuple[int, int], tuple[float | None, float | None]]:
    """(timed bytes, seconds when the row appeared) per (block size, batch size)."""

    timing: dict[tuple[int, int], tuple[float | None, float | None]] = {}
    rows = metadata.get("rows")
    if not isinstance(rows, list):
        return timing
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (_positive_int(row.get("block_size"), 0), _positive_int(row.get("batch_size"), 0))
        timing[key] = (_number(row.get("timed_bytes")), _number(row.get("seen_s")))
    return timing


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _process_observations(run_directory: Path, metadata: JsonObject) -> list[Observation]:
    """Observations of a run that used N single-thread pairs per sweep point.

    The pairs start their timed phases at slightly different moments, so adding
    their own averages overstates the aggregate: a pair that ran partly alone
    reports a rate the link only gave it for that part. Each pair's timed phase
    is therefore placed on a common clock from when its result row appeared
    (``row_seen_s``) and how long it ran (``timed_bytes`` over its rate), and the
    aggregate is the bytes all pairs moved over the span from the first start
    to the last end. It never exceeds N times the slowest pair's rate, and it
    equals the sum of the rates when the pairs ran together. ``overlap`` is the
    share of that span during which every pair was running, and ``skew`` the
    fastest rate over the slowest. Without timing records the aggregate is
    N times the slowest rate, which is only an upper bound.
    """

    processes = _positive_int(metadata.get("processes"), 0)
    points = metadata.get("points")
    if processes < 1 or not isinstance(points, list) or not points:
        raise ValueError("NIXL processes run metadata needs 'processes' and 'points'")
    run_id = _run_id(metadata, processes)
    observations = []
    for point_index, point in enumerate(points):
        if not isinstance(point, dict) or not isinstance(point.get("files"), list):
            raise ValueError("NIXL points must be objects with a 'files' list")
        block_size = _positive_int(point.get("block_size"), 0)
        batch_size = _positive_int(point.get("batch_size"), 0)
        rates = []
        initiator_files = []
        seen = []
        for entry in point["files"]:
            initiator_path = run_directory / _required_string(entry, "initiator_file")
            target_path = run_directory / _required_string(entry, "target_file")
            _require_files(initiator_path, target_path)
            rows = [
                row
                for row in _table_rows(initiator_path.read_text(encoding="utf-8", errors="replace"))
                if row[1] == block_size and row[2] == batch_size
            ]
            if len(rows) != 1:
                raise ValueError(
                    f"expected one block_size={block_size} batch_size={batch_size} row "
                    f"in {initiator_path}, found {len(rows)}"
                )
            rates.append(rows[0][3])
            initiator_files.append(initiator_path.name)
            seen.append(entry.get("row_seen_s"))
        if len(rates) != processes or min(rates) <= 0:
            raise ValueError(
                f"NIXL point block_size={block_size} batch_size={batch_size} needs "
                f"{processes} positive rates, got {rates}"
            )
        details: dict[str, JsonValue] = {
            "row_index": point_index,
            "initiator_file": initiator_files[0],
            "initiator_files": initiator_files,
            "process_rates_GB_s": rates,
            "skew": max(rates) / min(rates),
        }
        aggregate = processes * min(rates)
        timed_bytes = point.get("timed_bytes")
        if (
            isinstance(timed_bytes, (int, float))
            and not isinstance(timed_bytes, bool)
            and timed_bytes > 0
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in seen)
        ):
            ends = [float(value) for value in seen]  # type: ignore[arg-type]
            starts = [end - timed_bytes / (rate * 1e9) for end, rate in zip(ends, rates)]
            window = max(ends) - min(starts)
            aggregate = processes * timed_bytes / window / 1e9
            details["aggregate_method"] = "window"
            details["window_s"] = window
            details["overlap"] = max(0.0, min(ends) - max(starts)) / window
            details.update(
                cpu_metadata(run_directory, metadata, min(starts), max(ends))
            )
        else:
            details["aggregate_method"] = "slowest_rate"
        observations.append(
            _observation(aggregate, processes, block_size, batch_size, run_id, details)
        )
    return observations


def _observation(
    gb_per_second: float,
    threads: int,
    block_size: int,
    batch_size: int,
    run_id: str,
    metadata: dict[str, JsonValue],
) -> Observation:
    # NIXL reports decimal gigabytes per second; canonical throughput is bit/s.
    throughput = gb_per_second * 8e9
    if not math.isfinite(throughput):
        raise ValueError(f"non-finite NIXL throughput for run {run_id}")
    return Observation(
        metric="throughput",
        value=throughput,
        unit="bit/s",
        dimensions={
            "threads": threads,
            "block_size_bytes": block_size,
            "batch_size": batch_size,
        },
        sample_kind="table_row",
        run_id=run_id,
        metadata={"source_unit": "GB/s", **metadata},
    )


def _run_id(metadata: JsonObject, threads: int) -> str:
    run_number = _positive_int(metadata.get("run_number"), 1)
    return f"{_required_string(metadata, 'test_id')}-t{threads}-run{run_number}"


def _require_files(initiator_path: Path, target_path: Path) -> None:
    if not initiator_path.is_file():
        raise ValueError(f"NIXL initiator log not found: {initiator_path}")
    if not target_path.is_file():
        raise ValueError(f"NIXL target log not found: {target_path}")


def _table_rows(text: str) -> list[tuple[int, int, int, float]]:
    rows = []
    for row_index, line in enumerate(text.splitlines()):
        match = _ROW_RE.match(line)
        if match is not None:
            block_size, batch_size, gb_per_second = match.groups()
            rows.append((row_index, int(block_size), int(batch_size), float(gb_per_second)))
    return rows


def _load_object(path: Path) -> JsonObject:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read NIXL command metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return cast(JsonObject, value)


def _required_string(root: JsonObject, field: str) -> str:
    value = root.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _effective_threads(text: str, metadata: JsonObject) -> int:
    match = _THREADS_RE.search(text)
    if match is not None:
        return int(match.group(1))
    parameters = metadata.get("parameters")
    if isinstance(parameters, dict):
        value = parameters.get("num_threads", parameters.get("threads", 1))
        return _positive_int(value, 1)
    return 1


def _positive_int(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default
