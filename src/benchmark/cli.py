from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

from ..backends.registry import BackendRegistry, default_registry
from ..common.console import Colors, terminal_colors
from ..common.types import JsonValue
from .kubernetes import KubectlClient
from .artifacts import BenchmarkExecution
from .network import DEFAULT_INTERFACE, NetworkAttachment, SecondaryNetwork
from .contracts import (
    DEFAULT_IMAGE_PULL_POLICY,
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    IMAGE_PULL_POLICIES,
    BenchmarkConfig,
)
from .overwrite import OverwriteDeclined, confirm_overwrite
from .orchestrator import run_benchmark
from .profile import load_profile, profile_section
from .resources import PodResources
from .utils import parse_thread_spec


COMMON_PROFILE_FIELDS = frozenset(
    {"backend", "name", "test_id", "nodes", "dimensions", "execution", "network"}
)
NETWORK_SIDES = ("initiator", "target")
NETWORK_SIDE_FIELDS = frozenset({"attachment", "interface", "ip"})
COMMON_NODE_FIELDS = frozenset({"initiator", "target"})
COMMON_EXECUTION_FIELDS = frozenset(
    {
        "namespace",
        "image",
        "log_directory",
        "runs",
        "timeout_seconds",
        "keep_resources",
        "max_consecutive_failures",
        "image_pull_policy",
        "resources",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-benchmark.sh",
        description="Run a registered Kubernetes benchmark backend.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        metavar="FILE",
        help="Benchmark profile YAML file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing benchmark directory without asking.",
    )
    return parser


def parse_config(
    arguments: Sequence[str] | None = None,
    registry: BackendRegistry | None = None,
) -> BenchmarkConfig:
    args = build_parser().parse_args(arguments)
    profile = load_profile(args.profile)
    _reject_unknown_fields(profile, COMMON_PROFILE_FIELDS, "profile")
    nodes = profile_section(profile, "nodes")
    _reject_unknown_fields(nodes, COMMON_NODE_FIELDS, "profile.nodes")
    dimensions = profile_section(profile, "dimensions")
    execution = profile_section(profile, "execution")

    backend = _profile_string(profile, "backend")
    if backend is None:
        raise ValueError("profile field 'backend' is required")
    selected_registry = registry or default_registry()
    plugin = selected_registry.get(backend)
    network = _parse_network(profile, backend, plugin.supports_network)
    backend_parameters = profile_section(execution, backend)
    _reject_unknown_fields(
        execution,
        COMMON_EXECUTION_FIELDS | {backend},
        "profile.execution",
    )
    _reject_unknown_fields(
        dimensions,
        plugin.profile_dimensions,
        "profile.dimensions",
    )
    for old, new in plugin.renamed_profile_options.items():
        if old in backend_parameters:
            raise ValueError(f"profile.execution.{backend}.{old} was renamed: use {new}")
    _reject_unknown_fields(
        backend_parameters,
        plugin.profile_options,
        f"profile.execution.{backend}",
    )
    parameters: dict[str, JsonValue] = {}
    for name, value in backend_parameters.items():
        if name in plugin.profile_mapping_options:
            if value is not None and not isinstance(value, dict):
                raise ValueError(f"profile parameter {name!r} must be a mapping")
            for key, item in (value or {}).items():
                if not isinstance(key, str) or (
                    not isinstance(item, (str, int, float, bool)) or item is None
                ):
                    raise ValueError(
                        f"profile parameter {name!r} must map names to scalar values"
                    )
            parameters[name] = dict(value or {})  # type: ignore[assignment]
            continue
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise ValueError(f"profile parameter {name!r} must be a scalar")
        parameters[name] = value  # type: ignore[assignment]
    if backend == "nixl":
        if backend_parameters.get("iterations") is None:
            raise ValueError(
                "profile field 'execution.nixl.iterations' is required: it is nixlbench's "
                "iteration count (formerly execution.runs, which now repeats each thread "
                "count like for every other backend)"
            )
        _profile_positive_int(
            backend_parameters.get("iterations"), "execution.nixl.iterations", default=1
        )
        _profile_nonnegative_int(
            backend_parameters.get("warmup_iterations"),
            "warmup_iterations",
            required=True,
            location="execution.nixl",
        )

    initiator_node = _profile_string(nodes, "initiator")
    target_node = _profile_string(nodes, "target")
    if initiator_node is None or target_node is None:
        raise ValueError("profile must provide nodes.initiator and nodes.target")
    if initiator_node == target_node:
        raise ValueError("initiator and target nodes must be different")

    threads_spec = _profile_thread_spec(dimensions.get("threads"))
    thread_values = parse_thread_spec(threads_spec)
    parameters["threads"] = thread_values[0]
    parameters["threads_spec"] = threads_spec
    if backend == "nixl":
        for dimension_name in ("block_size", "batch_size"):
            start, maximum = _profile_range_bounds(
                dimensions.get(dimension_name), dimension_name
            )
            parameters[f"{dimension_name}_start"] = start
            parameters[f"{dimension_name}_max"] = maximum
    image = _profile_string(execution, "image")
    if image is None:
        raise ValueError("profile field 'execution.image' is required")
    namespace = _profile_string(execution, "namespace")
    kube_context, namespace = KubectlClient().context_and_namespace(namespace)
    log_directory = Path(_profile_string(execution, "log_directory") or "logs")
    name = _profile_string(profile, "name")
    test_id = _profile_string(profile, "test_id") or _default_test_id()
    runs = _profile_positive_int(execution.get("runs"), "runs", default=1)
    timeout = _profile_positive_int(
        execution.get("timeout_seconds"), "timeout_seconds", default=120
    )
    image_pull_policy = execution.get("image_pull_policy", DEFAULT_IMAGE_PULL_POLICY)
    if image_pull_policy not in IMAGE_PULL_POLICIES:
        raise ValueError(
            "profile field 'image_pull_policy' must be one of: "
            + ", ".join(IMAGE_PULL_POLICIES)
        )
    max_consecutive_failures = _profile_nonnegative_int(
        execution.get("max_consecutive_failures"),
        "max_consecutive_failures",
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
    )
    keep_resources = _profile_bool(
        execution.get("keep_resources"), "keep_resources", default=False
    )
    return BenchmarkConfig(
        backend=backend,
        test_id=test_id,
        initiator_node=initiator_node,
        target_node=target_node,
        namespace=namespace,
        image=image,
        log_directory=log_directory,
        name=name,
        kube_context=kube_context,
        runs=runs,
        timeout_seconds=timeout,
        keep_resources=keep_resources,
        max_consecutive_failures=max_consecutive_failures,
        image_pull_policy=image_pull_policy,
        network=network,
        resources=_parse_resources(execution),
        parameters=parameters,
    )


def main(
    arguments: Sequence[str] | None = None,
    registry: BackendRegistry | None = None,
) -> int:
    colors = terminal_colors(sys.stdout, sys.stderr)
    try:
        selected_registry = registry or default_registry()
        config = parse_config(arguments, selected_registry)
        force = build_parser().parse_args(arguments).force
        confirm_overwrite(config, colors=colors, force=force)
        execution = run_benchmark(
            config,
            selected_registry,
            colors=terminal_colors(sys.stdout, sys.stderr),
        )
    except OverwriteDeclined as error:
        print(f"{colors.yellow}{error}{colors.reset}", file=sys.stderr)
        return 1
    except (EOFError, OSError, RuntimeError, ValueError) as error:
        print(f"{colors.red}ERROR:{colors.reset} {error}", file=sys.stderr)
        return 2
    _print_summary(execution, colors)
    return 0 if execution.successful else 1


def _print_summary(execution: BenchmarkExecution, colors: Colors) -> None:
    failed = [outcome for outcome in execution.outcomes if not outcome.successful]
    succeeded = len(execution.outcomes) - len(failed)
    total = execution.planned_runs or len(execution.outcomes)
    parts = [f"{succeeded}/{total} runs succeeded"]
    if failed:
        parts.append(f"{len(failed)} failed")
    if execution.not_run:
        parts.append(f"{execution.not_run} not run")
    color = colors.green if execution.successful else colors.red
    print(f"{color}Benchmark summary: {', '.join(parts)}{colors.reset}")
    for outcome in failed:
        print(
            f"ERROR: threads={outcome.thread_count}, "
            f"run={outcome.run_number}: {outcome.error}",
            file=sys.stderr,
        )
    if execution.aborted_reason is not None:
        print(f"ERROR: {execution.aborted_reason}", file=sys.stderr)


def _parse_resources(execution: Mapping[str, object]) -> PodResources:
    section = profile_section(execution, "resources")
    _reject_unknown_fields(
        section, frozenset({"requests", "limits"}), "profile.execution.resources"
    )
    parsed: dict[str, dict[str, str]] = {}
    for kind in ("requests", "limits"):
        values = profile_section(section, kind)
        location = f"profile.execution.resources.{kind}"
        _reject_unknown_fields(values, frozenset({"cpu", "memory"}), location)
        parsed[kind] = {}
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError(f"{location}.{name} must be a string or number")
            parsed[kind][name] = str(value)
    return PodResources(requests=parsed["requests"], limits=parsed["limits"])


def _parse_network(
    profile: Mapping[str, object],
    backend: str,
    supported: bool,
) -> SecondaryNetwork | None:
    if "network" not in profile or profile["network"] is None:
        return None
    if not supported:
        raise ValueError(
            f"backend {backend!r} does not support the 'network' section yet"
        )
    section = profile_section(profile, "network")
    _reject_unknown_fields(section, frozenset(NETWORK_SIDES), "profile.network")
    sides: dict[str, NetworkAttachment] = {}
    for side in NETWORK_SIDES:
        if side not in section:
            raise ValueError(f"profile section 'network.{side}' is required")
        fields = profile_section(section, side)
        location = f"profile.network.{side}"
        _reject_unknown_fields(fields, NETWORK_SIDE_FIELDS, location)
        attachment = _profile_string(fields, "attachment")
        if attachment is None:
            raise ValueError(f"profile field '{location}.attachment' is required")
        try:
            sides[side] = NetworkAttachment(
                attachment=attachment,
                interface=_profile_string(fields, "interface") or DEFAULT_INTERFACE,
                ip=_profile_string(fields, "ip"),
            )
        except ValueError as error:
            raise ValueError(f"{location}: {error}") from error
    return SecondaryNetwork(**sides)


def _reject_unknown_fields(
    section: Mapping[str, object],
    allowed: frozenset[str],
    location: str,
) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(
            f"unsupported field(s) in {location}: {', '.join(unknown)}"
        )


def _profile_string(
    section: Mapping[str, object],
    name: str,
) -> str | None:
    value = section.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"profile field {name!r} must be a non-empty string")
    return value


def _profile_range_bounds(value: object, name: str) -> tuple[int, int]:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(
            f"profile field 'dimensions.{name}' must be an integer or range"
        )
    parts = str(value).split(":")
    if len(parts) not in (1, 2):
        raise ValueError(
            f"profile field 'dimensions.{name}' must use VALUE or START:MAX"
        )
    try:
        numbers = [_parse_dimension_number(part, name) for part in parts]
    except ValueError as error:
        raise ValueError(
            f"profile field 'dimensions.{name}' must use valid positive values"
        ) from error
    start = numbers[0]
    maximum = start if len(numbers) == 1 else numbers[1]
    if start < 1 or maximum < start:
        raise ValueError(
            f"profile field 'dimensions.{name}' must use positive values with start <= max"
        )
    if name in {"block_size", "batch_size"}:
        if not _is_power_of_two(start) or not _is_power_of_two(maximum):
            raise ValueError(
                f"profile field 'dimensions.{name}' must use powers of two"
            )
        ratio = maximum // start
        if maximum % start != 0 or not _is_power_of_two(ratio):
            raise ValueError(
                f"profile field 'dimensions.{name}' must increase by powers of two"
            )
    return start, maximum


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _parse_dimension_number(value: str, name: str) -> int:
    if name != "block_size":
        if not value.isdigit():
            raise ValueError(value)
        return int(value)
    suffix = value[-1:].upper()
    multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}.get(suffix, 1)
    number = value[:-1] if suffix in {"K", "M", "G"} else value
    if not number.isdigit():
        raise ValueError(value)
    return int(number) * multiplier


def _profile_thread_spec(value: object) -> str:
    if value is None:
        return "1"
    if isinstance(value, list):
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError("profile field 'threads' lists must contain integers")
        value = ",".join(str(item) for item in value)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("profile field 'threads' must be a string, integer or list of integers")
    text = str(value)
    if not text:
        raise ValueError("profile field 'threads' must not be empty")
    return text


def _profile_nonnegative_int(
    value: object,
    name: str,
    *,
    required: bool = False,
    default: int = 0,
    location: str = "execution",
) -> int:
    if value is None:
        if required:
            raise ValueError(f"profile field '{location}.{name}' is required")
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"profile field {name!r} must be a non-negative integer")
    return value


def _profile_positive_int(
    value: object,
    name: str,
    *,
    default: int,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"profile field {name!r} must be a positive integer")
    return value


def _profile_bool(
    value: object,
    name: str,
    *,
    default: bool,
) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"profile field {name!r} must be boolean")
    return value


def _default_test_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"test-{timestamp}-{os.getpid()}"
