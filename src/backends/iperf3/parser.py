from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, cast

from ...benchmark.cpu import cpu_metadata
from ...common.types import JsonValue
from ...graphing.contracts import Observation


JsonObject = Mapping[str, object]
MIN_INTERVAL_SECONDS = 0.5
MAX_INTERVAL_MISMATCH_SECONDS = 0.1


class Iperf3Parser:
    """Parse iperf3 JSON while preserving interval-level observations."""

    def discover(self, raw_directory: Path) -> tuple[Path, ...]:
        run_directories = {
            command_file.parent
            for command_file in raw_directory.rglob("*-command.json")
        }
        if not run_directories:
            raise ValueError(f"no iperf3 command metadata found in {raw_directory}")
        return tuple(sorted(run_directories))

    def parse(self, run_directory: Path) -> tuple[Observation, ...]:
        command_files = sorted(run_directory.glob("*-command.json"))
        if not command_files:
            raise ValueError(f"no iperf3 command metadata found in {run_directory}")

        observations: list[Observation] = []
        for command_file in command_files:
            metadata = _load_object(command_file)
            client_path = _referenced_file(
                run_directory,
                _required_string(metadata, "client_file"),
            )
            server_path = _referenced_file(
                run_directory,
                _required_string(metadata, "server_file"),
            )
            if not client_path.is_file():
                raise ValueError(f"client result not found: {client_path}")
            if not server_path.is_file():
                raise ValueError(f"server log not found: {server_path}")

            client = _load_object(client_path)
            client_status = metadata.get("client_status")
            if client_status != 0:
                raise ValueError(
                    f"iperf3 client exited with status {client_status} in {client_path}"
                )
            error = client.get("error")
            if isinstance(error, str) and error:
                raise ValueError(f"iperf3 reported an error: {error}")

            parameters = _parameters(metadata)
            threads = _thread_count(client, parameters)
            dimensions: dict[str, int] = {"threads": threads}
            dimension_labels: dict[str, str] = {"threads": str(threads)}

            server_output = _object(client.get("server_output_json"), "server_output_json")
            intervals = server_output.get("intervals")
            if not isinstance(intervals, list):
                raise ValueError("expected server_output_json.intervals to be a list")

            test_id = _required_string(metadata, "test_id")
            run_number = _positive_or_default(metadata.get("run_number"), 1)
            run_id = f"{test_id}-t{threads}-run{run_number}"
            cpu = _run_cpu(run_directory, metadata, parameters)
            for index, interval_value in enumerate(cast(list[object], intervals)):
                interval = _object(interval_value, f"intervals[{index}]")
                summary = _object(interval.get("sum"), f"intervals[{index}].sum")
                bits_per_second = _number(summary.get("bits_per_second"))
                if bits_per_second is None:
                    continue
                start = _number(summary.get("start"))
                end = _number(summary.get("end"))
                seconds = _number(summary.get("seconds"))
                omitted = summary.get("omitted") is True
                valid = (
                    not omitted
                    and start is not None
                    and end is not None
                    and seconds is not None
                    and end >= start
                    and seconds >= MIN_INTERVAL_SECONDS
                    and abs(seconds - (end - start))
                    <= MAX_INTERVAL_MISMATCH_SECONDS
                )
                interval_metadata: dict[str, JsonValue] = {
                    "interval_index": index,
                    "start_seconds": start,
                    "end_seconds": end,
                    "seconds": seconds,
                    "omitted": omitted,
                    "valid": valid,
                    "dimension_labels": dimension_labels,
                    **cpu,
                }
                observations.append(
                    Observation(
                        metric="throughput",
                        value=bits_per_second,
                        unit="bit/s",
                        dimensions=dimensions,
                        sample_kind="interval",
                        run_id=run_id,
                        metadata=interval_metadata,
                    )
                )
        return tuple(observations)


def _run_cpu(
    run_directory: Path,
    metadata: JsonObject,
    parameters: Mapping[str, object],
) -> dict[str, JsonValue]:
    """Each side's CPU over the measured part of the run (its last duration_seconds)."""

    finished = _number(metadata.get("client_finished_s"))
    duration = _number(parameters.get("duration_seconds", 10))
    if finished is None or duration is None or duration <= 0:
        return {}
    return cpu_metadata(run_directory, metadata, finished - duration, finished)


def _load_object(path: Path) -> JsonObject:
    try:
        text = path.read_text(encoding="utf-8")
        try:
            value: object = json.loads(text)
        except json.JSONDecodeError:
            stripped = text.lstrip()
            value, end = json.JSONDecoder().raw_decode(stripped)
            stripped[end:].strip()
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return cast(JsonObject, value)


def _object(value: object, field: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"expected {field} to be a JSON object")
    return cast(JsonObject, value)


def _required_string(root: JsonObject, field: str) -> str:
    value = root.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _parameters(metadata: JsonObject) -> Mapping[str, object]:
    value = metadata.get("parameters", {})
    if not isinstance(value, dict):
        raise ValueError("parameters must be a JSON object")
    return value


def _thread_count(client: JsonObject, parameters: Mapping[str, object]) -> int:
    start = _object(client.get("start", {}), "start")
    test_start = _object(start.get("test_start", {}), "start.test_start")
    value = test_start.get("num_streams", parameters.get("threads", 1))
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("iperf3 thread count must be a positive integer")
    return value


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _referenced_file(run_directory: Path, reference: str) -> Path:
    path = (run_directory / reference).resolve()
    try:
        path.relative_to(run_directory.parent.parent.resolve())
    except ValueError as error:
        raise ValueError(f"raw file reference escapes test directory: {reference}") from error
    return path


def _positive_or_default(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default
