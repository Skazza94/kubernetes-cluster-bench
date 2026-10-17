from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from ...benchmark.artifacts import write_json
from ...benchmark.contracts import (
    BackendResources,
    BenchmarkConfig,
    BenchmarkRunner,
    RunContext,
)
from ...benchmark.cpu import ROLES as CPU_ROLES, CpuSampler
from ...benchmark.kubernetes import KubectlClient
from ...benchmark.manifest import dump_manifest
from ...benchmark.network import (
    apply_secondary_network,
    discover_addresses,
    set_container_env,
    verify_attachments,
)
from ...benchmark.process import terminate_process
from ...benchmark.resources import apply_resources
from ...benchmark.utils import (
    format_eta,
    parse_thread_spec,
    resource_overrides,
    utc_timestamp,
)
from ...common.console import Colors
from ...common.types import JsonValue


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "nixl-template.yaml"
DEFAULT_RUN_TIMEOUT_SECONDS = 1800
GPU_RESOURCE = "nvidia.com/gpu"
CPU_UCX_TLS = "tcp,self"
ASIO_BASE_PORT = 12345  # each nixlbench pair of a run rendezvous on its own port, counting up
PARALLELISM_MODES = ("threads", "processes")

# nixlbench flags that can be set from the profile (execution.nixl.args).
INT_ARGS = {
    "pipeline_depth": 1,
    "large_blk_iter_ftr": 1,
    "progress_threads": 0,
}
BOOL_ARGS = frozenset(
    {
        "enable_pt",
        "prepared_xfer",
        "recreate_xfer",
        "reregister_mem",
        "use_hugepages",
        "enable_vmm",
        "check_consistency",
    }
)
CHOICE_ARGS = {"op_type": ("READ", "WRITE")}

# Flags the runner derives from the profile; setting them here would conflict.
MANAGED_ARGS = {
    "runtime_type": "fixed to ASIO",
    "asio_address": "derived from the initiator pod",
    "asio_port": "assigned per nixlbench pair",
    "backend": "set with execution.nixl.backend",
    "initiator_seg_type": "fixed to DRAM",
    "target_seg_type": "fixed to DRAM",
    "total_buffer_size": "computed from dimensions.block_size/batch_size/threads",
    "start_block_size": "set with dimensions.block_size",
    "max_block_size": "set with dimensions.block_size",
    "start_batch_size": "set with dimensions.batch_size",
    "max_batch_size": "set with dimensions.batch_size",
    "num_threads": "set with dimensions.threads",
    "num_iter": "set with execution.nixl.iterations",
    "warmup_iter": "set with execution.nixl.warmup_iterations",
}

LARGE_BLOCK_SIZE = 1 << 20  # nixlbench reduces iterations for blocks above 1 MiB
DEFAULT_LARGE_BLK_ITER_FTR = 16

# NIXL backends (plugins) the benchmark can drive, over TCP. Their names are the
# values nixlbench takes for --backend.
BACKENDS = ("UCX", "LIBFABRIC", "Mooncake")
DEFAULT_BACKEND = "UCX"
# Besides NIXL_ and OMP_, each backend's own variables may be set in execution.nixl.env.
ENV_PREFIX = {"UCX": "UCX", "LIBFABRIC": "FI", "Mooncake": "MC"}
# Set by default so that the backend stays on TCP even if the pod sees RDMA devices.
DEFAULT_ENV = {
    "UCX": {},
    "LIBFABRIC": {"FI_PROVIDER": "tcp"},
    "Mooncake": {"MC_FORCE_TCP": "1"},
}
# What pins each backend to the network section's interface. Set from the profile.
NETWORK_MANAGED_ENV = {
    "UCX": "UCX_NET_DEVICES",
    "LIBFABRIC": "FI_TCP_IFACE",
    "Mooncake": "NIXL_MOONCAKE_IP_ADDR",
}
MOONCAKE_IO_THREADS = "MC_NUM_TCP_IO_THREADS"
# The Mooncake plugin has no progress thread support.
UNSUPPORTED_ARGS = {"Mooncake": ("enable_pt", "progress_threads")}
# Flags a backend cannot work without. The LIBFABRIC plugin completes its
# connection handshake in its progress thread, so without --enable_pt the
# target never answers and the run hangs until the timeout.
REQUIRED_ARGS: dict[str, dict[str, object]] = {"LIBFABRIC": {"enable_pt": True}}
UNUSED_UCX_VARIABLE = re.compile(r"unused environment variable: (\S+)")
WARN_UNUSED = "UCX_WARN_UNUSED_ENV_VARS"


class NixlRunner(BenchmarkRunner):
    """Run NIXLBench initiator and target processes on two GPU pods."""

    name = "nixl"

    def __init__(
        self,
        manifest_template: Path = DEFAULT_MANIFEST,
        kubectl: KubectlClient | None = None,
    ) -> None:
        self.manifest_template = manifest_template
        self.kubectl = kubectl or KubectlClient()
        self._processes: dict[str, list[subprocess.Popen[bytes]]] = {}
        self._colors: Colors | None = None

    def validate(self, config: BenchmarkConfig) -> None:
        if not self.manifest_template.is_file():
            raise ValueError(f"NIXL manifest not found: {self.manifest_template}")
        iterations = config.parameters.get("iterations")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("NIXL iterations must be a positive integer")
        warmup = config.parameters.get("warmup_iterations")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError("NIXL warmup_iterations must be a non-negative integer")
        run_timeout = config.parameters.get(
            "run_timeout_seconds", DEFAULT_RUN_TIMEOUT_SECONDS
        )
        if (
            isinstance(run_timeout, bool)
            or not isinstance(run_timeout, int)
            or run_timeout < 1
        ):
            raise ValueError("NIXL run_timeout_seconds must be a positive integer")
        if not isinstance(config.parameters.get("gpu", True), bool):
            raise ValueError("NIXL gpu must be true or false")
        if _parallelism(config) not in PARALLELISM_MODES:
            raise ValueError(
                "NIXL parallelism must be one of: " + ", ".join(PARALLELISM_MODES)
            )
        backend = _backend(config)
        if backend not in BACKENDS:
            raise ValueError("NIXL backend must be one of: " + ", ".join(BACKENDS))
        if config.network is None:
            for name, setting in _mapping(config.parameters.get("env"), "env").items():
                if isinstance(setting, str) and "{address}" in setting:
                    raise ValueError(
                        f"NIXL env.{name} uses {{address}}, which needs a network section"
                    )
        _validate_args(config.parameters.get("args"), backend=backend)
        _validate_env(
            config.parameters.get("env"),
            network_set=config.network is not None,
            backend=backend,
        )
        self._validate_pipeline_depth(config)

    def _validate_pipeline_depth(self, config: BenchmarkConfig) -> None:
        args = config.parameters.get("args") or {}
        depth = args.get("pipeline_depth", 1)
        spec = config.parameters.get("threads_spec")
        block_max = config.parameters.get("block_size_max")
        warmup = config.parameters.get("warmup_iterations")
        if depth <= 1 or not isinstance(spec, str) or not isinstance(block_max, int) or not isinstance(warmup, int):
            return
        _check_pipeline_depth(
            depth=depth,
            iterations=_iterations(config),
            warmup_iterations=warmup,
            # Each process of the "processes" mode runs a single nixlbench thread.
            thread_values=(1,) if _parallelism(config) == "processes" else parse_thread_spec(spec),
            max_block_size=block_max,
            large_blk_iter_ftr=args.get("large_blk_iter_ftr", DEFAULT_LARGE_BLK_ITER_FTR),
        )

    def render_manifest(self, config: BenchmarkConfig, destination: Path) -> None:
        self.validate(config)
        manifest = self.manifest_template.read_text(encoding="utf-8")
        replacements = {
            "__TEST_ID__": config.test_id,
            "__CLIENT_NODE__": config.target_node,
            "__SERVER_NODE__": config.initiator_node,
            "__IMAGE__": config.image,
            "__IMAGE_PULL_POLICY__": config.image_pull_policy,
        }
        missing = [marker for marker in replacements if marker not in manifest]
        if missing:
            raise ValueError(
                "NIXL manifest is missing required placeholder(s): "
                + ", ".join(missing)
            )
        for marker, value in replacements.items():
            manifest = manifest.replace(marker, value)
        remaining = [marker for marker in replacements if marker in manifest]
        if remaining:
            raise ValueError(
                "NIXL manifest still contains placeholder(s): "
                + ", ".join(remaining)
            )
        if not config.parameters.get("gpu", True):
            manifest = _without_gpu(manifest)
        if config.network is not None:
            names = self.resource_names(config)
            roles = {names["initiator_pod"]: "initiator", names["target_pod"]: "target"}
            manifest = apply_secondary_network(manifest, config.network, roles)
            if _backend(config) == "UCX":
                # UCX would otherwise be free to carry the data over eth0. The other
                # backends are pinned when nixlbench starts (see _environment).
                manifest = set_container_env(
                    manifest,
                    {
                        pod: {"UCX_NET_DEVICES": config.network.for_role(role).interface}
                        for pod, role in roles.items()
                    },
                )
        if not config.resources.is_empty:
            manifest = apply_resources(manifest, config.resources)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(manifest, encoding="utf-8")

    def deploy(self, config: BenchmarkConfig, manifest: Path) -> BackendResources:
        self.validate(config)
        names = self.resource_names(config)
        if config.network is not None:
            verify_attachments(self.kubectl, config.namespace, config.network)
        for resource in names["collision_targets"].split(","):
            if self.kubectl.resource_exists(resource, config.namespace):
                raise ValueError(
                    f"resource already exists: {resource}; retry with a clean namespace"
                )
        try:
            self.kubectl.apply_manifest(config.namespace, manifest)
            for pod in (names["initiator_pod"], names["target_pod"]):
                self.kubectl.wait_for(
                    config.namespace,
                    f"pod/{pod}",
                    "jsonpath={.status.phase}=Running",
                    config.timeout_seconds,
                )
            addresses: dict[str, str] = {}
            if config.network is not None:
                # The ASIO rendezvous must happen on the secondary network too.
                addresses = discover_addresses(
                    self.kubectl,
                    config.namespace,
                    config.network,
                    {
                        "initiator": names["initiator_pod"],
                        "target": names["target_pod"],
                    },
                )
                address = addresses["initiator_address"]
            else:
                address = self.kubectl.output(
                    [
                        "get",
                        "pod",
                        names["initiator_pod"],
                        "-n",
                        config.namespace,
                        "-o",
                        "jsonpath={.status.podIP}",
                    ]
                )
            if not address:
                raise ValueError("unable to determine the NIXL initiator pod IP")
        except BaseException:
            self.kubectl.delete_manifest(config.namespace, manifest)
            raise
        return BackendResources(
            names={
                "initiator_pod": names["initiator_pod"],
                "target_pod": names["target_pod"],
                "initiator_ip": address,
                **addresses,
            },
            manifest_path=manifest,
        )

    def execute(
        self,
        config: BenchmarkConfig,
        resources: BackendResources,
        output_directory: Path,
        context: RunContext | None = None,
    ) -> Path:
        self.validate(config)
        self._colors = context.colors if context is not None else self._colors
        self._report_eta(config, context)
        output_directory.mkdir(parents=True, exist_ok=True)
        thread_count = int(config.parameters.get("threads", 1))
        run_number = int(config.parameters.get("run_number", 1))
        mode = _parallelism(config)
        stem = f"{config.test_id}-t{thread_count}-run{run_number}"
        command_log = output_directory / f"{stem}-command.json"
        # "threads": one nixlbench pair running the whole sweep with N threads.
        # "processes": N single-thread pairs per (block size, batch size) point,
        # so every point is measured while all of them transfer together.
        pairs = thread_count if mode == "processes" else 1
        threads_per_pair = 1 if mode == "processes" else thread_count
        parameters = dict(config.parameters)
        parameters["num_threads"] = threads_per_pair
        iterations = _iterations(config)
        warmup = int(parameters["warmup_iterations"])
        parameters["num_iter"] = iterations
        parameters["warmup_iter"] = warmup
        parameters["parallelism"] = mode
        backend = str(_backend(config))
        parameters["backend"] = backend
        block_start = int(parameters["block_size_start"])
        block_max = int(parameters["block_size_max"])
        batch_start = int(parameters["batch_size_start"])
        batch_max = int(parameters["batch_size_max"])
        parameters["total_buffer_size"] = _minimum_total_buffer_size(
            threads_per_pair, block_max, batch_max
        )
        extra_flags = _extra_flags(
            {**REQUIRED_ARGS.get(backend, {}), **(config.parameters.get("args") or {})}
        )
        env = config.parameters.get("env")
        env_prefixes = {
            role: _env_prefix(_environment(config, resources, role), warn_unused=backend == "UCX" and bool(env))
            for role in ("initiator", "target")
        }
        if mode == "processes":
            sweep = [
                (block, block, batch, batch)
                for block in _powers_of_two(block_start, block_max)
                for batch in _powers_of_two(batch_start, batch_max)
            ]
        else:
            sweep = [(block_start, block_max, batch_start, batch_max)]

        iter_factor = int(
            (config.parameters.get("args") or {}).get("large_blk_iter_ftr", DEFAULT_LARGE_BLK_ITER_FTR)
        )

        # Plan every nixlbench execution up front: each gets its own ASIO port.
        points: list[dict[str, JsonValue]] = []
        port = ASIO_BASE_PORT
        for start_block, end_block, start_batch, end_batch in sweep:
            label = f"-b{end_block}-n{end_batch}" if mode == "processes" else ""
            files: list[dict[str, JsonValue]] = []
            for index in range(1, pairs + 1):
                suffix = f"{label}-p{index}" if mode == "processes" else ""
                args = self._nixl_args(
                    threads_per_pair,
                    iterations,
                    warmup,
                    _minimum_total_buffer_size(threads_per_pair, end_block, end_batch),
                    start_block,
                    end_block,
                    start_batch,
                    end_batch,
                    resources.names["initiator_ip"],
                    asio_port=port,
                    backend=backend,
                )
                files.append({
                    "asio_port": port,
                    "initiator_file": f"{stem}{suffix}-initiator.log",
                    "target_file": f"{stem}{suffix}-target.log",
                    "args": [*args, *extra_flags],
                })
                port += 1
            point: dict[str, JsonValue] = {
                "block_size": end_block,
                "batch_size": end_batch,
                "files": files,
            }
            if mode == "processes":
                point["timed_bytes"] = _timed_bytes(
                    iterations, iter_factor, threads_per_pair, end_block, end_batch
                )
            points.append(point)

        metadata: dict[str, JsonValue] = {
            "backend": self.name,
            "test_id": config.test_id,
            "run_number": run_number,
            "kube_context": config.kube_context,
            "initiator_node": config.initiator_node,
            "target_node": config.target_node,
            "namespace": config.namespace,
            "image": config.image,
            "parameters": parameters,
            "network": self._network_metadata(config, resources),
            "resource_overrides": resource_overrides(config),
            "nixl_args": points[0]["files"][0]["args"],  # type: ignore[index]
            "nixl_env": dict(env) if env else {},
            "nixl_backend": backend,
            "nixl_env_applied": {
                role: {name: str(value) for name, value in _environment(config, resources, role).items()}
                for role in ("initiator", "target")
            },
            "parallelism": mode,
            "started_at": utc_timestamp(),
        }
        if mode == "processes":
            metadata["processes"] = pairs
            metadata["points"] = [
                {
                    "block_size": point["block_size"],
                    "batch_size": point["batch_size"],
                    "timed_bytes": point["timed_bytes"],
                    "files": [
                        {key: entry[key] for key in ("asio_port", "initiator_file", "target_file")}
                        for entry in point["files"]  # type: ignore[union-attr]
                    ],
                }
                for point in points
            ]
        else:
            first = points[0]["files"][0]  # type: ignore[index]
            metadata["initiator_file"] = first["initiator_file"]
            metadata["target_file"] = first["target_file"]
            # Each row's end time and bytes place its timed phase on the CPU samples' clock.
            metadata["rows"] = [
                {
                    "block_size": block,
                    "batch_size": batch,
                    "timed_bytes": _timed_bytes(iterations, iter_factor, threads_per_pair, block, batch),
                }
                for block, batch in _expected_rows(sweep[0])
            ]
        cpu_files = {role: f"{stem}-cpu-{role}.log" for role in CPU_ROLES}
        thread_files = {role: f"{stem}-threads-{role}.log" for role in CPU_ROLES}
        metadata["cpu_files"] = dict(cpu_files)
        metadata["thread_files"] = dict(thread_files)
        write_json(command_log, metadata)

        origin = time.monotonic()
        sampler = CpuSampler(
            self.kubectl,
            config.namespace,
            {role: resources.names[f"{role}_pod"] for role in CPU_ROLES},
            {role: output_directory / name for role, name in cpu_files.items()},
            origin,
            thread_files={role: output_directory / name for role, name in thread_files.items()},
        )
        with sampler:
            return self._execute_points(
                config,
                resources,
                context,
                output_directory,
                metadata,
                command_log,
                points,
                sweep,
                env_prefixes,
                origin,
            )

    def _execute_points(
        self,
        config: BenchmarkConfig,
        resources: BackendResources,
        context: RunContext | None,
        output_directory: Path,
        metadata: dict[str, JsonValue],
        command_log: Path,
        points: list[dict[str, JsonValue]],
        sweep: list[tuple[int, int, int, int]],
        env_prefixes: Mapping[str, Sequence[str]],
        origin: float,
    ) -> Path:
        mode = metadata["parallelism"]
        pairs = len(points[0]["files"])  # type: ignore[arg-type]
        env = config.parameters.get("env")
        backend = metadata["nixl_backend"]
        run_timeout = int(
            config.parameters.get("run_timeout_seconds", DEFAULT_RUN_TIMEOUT_SECONDS)
        )
        initiator_failure = target_failure = 0
        logs: list[Path] = []
        for number, point in enumerate(points, start=1):
            files = point["files"]  # type: ignore[assignment]
            initiator_logs = [output_directory / str(entry["initiator_file"]) for entry in files]
            target_logs = [output_directory / str(entry["target_file"]) for entry in files]
            logs.extend([*initiator_logs, *target_logs])
            where = ""
            if mode == "processes":
                where = (
                    f" at block_size={_format_size(int(point['block_size']))}"  # type: ignore[arg-type]
                    f" batch_size={point['batch_size']}"
                )
                self._report(
                    context,
                    f"NIXL point {number}/{len(points)}: {pairs} process(es){where}",
                )
            try:
                statuses, row_times = self._run_group(
                    config,
                    resources,
                    context,
                    env_prefixes,
                    [(entry["args"], log, target) for entry, log, target in zip(files, initiator_logs, target_logs)],
                    run_timeout,
                    config.test_id,
                    origin,
                    [_expected_rows(sweep[number - 1])] * len(files),
                )
            except subprocess.TimeoutExpired:
                metadata["timed_out"] = True
                metadata["finished_at"] = utc_timestamp()
                write_json(command_log, metadata)
                raise RuntimeError(
                    f"NIXLBench timed out after {run_timeout}s (run_timeout_seconds){where}"
                ) from None
            # When each result row appeared, for the aggregate window and the CPU samples.
            if mode == "processes":
                row = (int(point["block_size"]), int(point["batch_size"]))  # type: ignore[arg-type]
                for entry, seen in zip(metadata["points"][number - 1]["files"], row_times):  # type: ignore[index]
                    entry["row_seen_s"] = seen.get(row)
            else:
                for entry in metadata["rows"]:  # type: ignore[union-attr]
                    entry["seen_s"] = row_times[0].get((entry["block_size"], entry["batch_size"]))  # type: ignore[index]
            tolerated = _tolerated_target_exits(
                statuses,
                initiator_logs,
                target_logs,
                _expected_rows(sweep[number - 1]),
            )
            if tolerated:
                for entry in tolerated:
                    entry["point"] = where.strip() or "whole sweep"
                metadata.setdefault("tolerated_target_exits", []).extend(tolerated)  # type: ignore[union-attr]
                self._report(
                    context,
                    f"warning: {len(tolerated)} target(s) failed at shutdown after their "
                    f"initiator finished; the results are complete{where}",
                )
                statuses = {name: 0 for name in statuses}
            if any(status != 0 for status in statuses.values()):
                initiator_failure = next(
                    (s for n, s in statuses.items() if n.startswith("initiator") and s != 0), 0
                )
                target_failure = next(
                    (s for n, s in statuses.items() if n.startswith("target") and s != 0), 0
                )
                metadata["initiator_status"] = initiator_failure
                metadata["target_status"] = target_failure
                metadata["finished_at"] = utc_timestamp()
                write_json(command_log, metadata)
                reason = _first_error_line(*initiator_logs, *target_logs)
                summary = ", ".join(f"{name}={status}" for name, status in statuses.items())
                raise RuntimeError(
                    f"NIXLBench failed ({summary}){where}" + (f": {reason}" if reason else "")
                )
        metadata["initiator_status"] = 0
        metadata["target_status"] = 0
        metadata["finished_at"] = utc_timestamp()
        write_json(command_log, metadata)
        if env and backend == "UCX":
            unknown = _unknown_variables(
                *(log.read_text(encoding="utf-8", errors="replace") for log in logs)
            )
            if unknown:
                raise RuntimeError(
                    "UCX does not know the variable(s) " + ", ".join(unknown)
                    + "; check the names in execution.nixl.env"
                )
        return output_directory

    def _run_group(
        self,
        config: BenchmarkConfig,
        resources: BackendResources,
        context: RunContext | None,
        env_prefixes: Mapping[str, Sequence[str]],
        jobs: Sequence[tuple[Sequence[str], Path, Path]],
        run_timeout: int,
        process_key: str,
        origin: float,
        watch_rows: Sequence[Sequence[tuple[int, int]]],
    ) -> tuple[dict[str, int], list[dict[tuple[int, int], float]]]:
        """Run one or more nixlbench pairs side by side and wait for all of them.

        Each job is (nixlbench arguments, initiator log, target log). The
        initiators start first and must be ready before any target connects.
        Raises subprocess.TimeoutExpired, after stopping everything, if the
        pairs run longer than ``run_timeout``.

        Returns the exit statuses and, for each job, the seconds after
        ``origin`` at which each of its ``watch_rows`` (block size, batch size)
        result rows first appeared in its initiator log.
        """

        started = self._processes.setdefault(process_key, [])
        watcher = _RowWatcher([log for _, log, _ in jobs], watch_rows, origin)
        initiators = []
        for args, initiator_log, _ in jobs:
            process = self.kubectl.start_exec(
                config.namespace,
                resources.names["initiator_pod"],
                [*env_prefixes["initiator"], "nixlbench", *args],
                initiator_log,
            )
            started.append(process)
            initiators.append(process)
        for process, (_, initiator_log, _) in zip(initiators, jobs):
            self._wait_for_initiator(config, process, initiator_log)
        self._report(
            context,
            "NIXL initiator ready" if len(jobs) == 1 else f"{len(jobs)} NIXL initiators ready",
        )
        targets = []
        for args, _, target_log in jobs:
            process = self.kubectl.start_exec(
                config.namespace,
                resources.names["target_pod"],
                [*env_prefixes["target"], "nixlbench", *args],
                target_log,
            )
            started.append(process)
            targets.append(process)
        names: dict[str, subprocess.Popen[bytes]] = {}
        for index, (initiator, target) in enumerate(zip(initiators, targets), start=1):
            label = f" {index}" if len(jobs) > 1 else ""
            names[f"initiator{label}"] = initiator
            names[f"target{label}"] = target
        watcher.start()
        try:
            statuses = _wait_for_all(names, run_timeout)
        except subprocess.TimeoutExpired:
            for process in reversed(started):
                terminate_process(process)
            raise
        finally:
            watcher.stop()
        return statuses, watcher.times

    def cleanup(
        self,
        config: BenchmarkConfig,
        resources: BackendResources | None,
    ) -> None:
        for process in reversed(self._processes.pop(config.test_id, [])):
            terminate_process(process)
        if resources is None or config.keep_resources or resources.manifest_path is None:
            return
        self.kubectl.delete_manifest(config.namespace, resources.manifest_path)

    def _network_metadata(
        self,
        config: BenchmarkConfig,
        resources: BackendResources,
    ) -> dict[str, JsonValue] | None:
        if config.network is None:
            return None
        return {
            role: {
                "attachment": config.network.for_role(role).attachment,
                "interface": config.network.for_role(role).interface,
                "address": resources.names.get(f"{role}_address"),
            }
            for role in ("initiator", "target")
        }

    def resource_names(self, config: BenchmarkConfig) -> dict[str, str]:
        initiator = f"nixlbench-server-{config.test_id}"
        target = f"nixlbench-client-{config.test_id}"
        return {
            "initiator_pod": initiator,
            "target_pod": target,
            "collision_targets": f"pod/{initiator},pod/{target}",
        }

    def _nixl_args(
        self,
        thread_count: int,
        iteration_count: int,
        warmup_count: int,
        total_buffer_size: int,
        block_size_start: int,
        block_size_max: int,
        batch_size_start: int,
        batch_size_max: int,
        initiator_ip: str,
        asio_port: int = ASIO_BASE_PORT,
        backend: str = DEFAULT_BACKEND,
    ) -> list[str]:
        return [
            "--runtime_type", "ASIO",
            "--asio_address", initiator_ip,
            "--asio_port", str(asio_port),
            "--backend", backend,
            "--initiator_seg_type", "DRAM",
            "--target_seg_type", "DRAM",
            "--total_buffer_size", str(total_buffer_size),
            "--start_block_size", str(block_size_start),
            "--max_block_size", str(block_size_max),
            "--start_batch_size", str(batch_size_start),
            "--max_batch_size", str(batch_size_max),
            "--num_threads", str(thread_count),
            "--num_iter", str(iteration_count),
            "--warmup_iter", str(warmup_count),
        ]

    def _wait_for_initiator(
        self,
        config: BenchmarkConfig,
        process: subprocess.Popen[bytes],
        log_path: Path,
    ) -> None:
        deadline = time.monotonic() + config.timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                reason = _first_error_line(log_path)
                raise RuntimeError(
                    "NIXL initiator exited before becoming ready"
                    + (f": {reason}" if reason else "")
                )
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            if "Using address " in text and "port " in text:
                return
            time.sleep(0.2)
        raise TimeoutError("timed out waiting for NIXL initiator readiness")

    def _report_eta(
        self,
        config: BenchmarkConfig,
        context: RunContext | None,
    ) -> None:
        if context is None:
            return
        if context.completed_runs > 0:
            elapsed = time.monotonic() - context.benchmark_started_monotonic
            eta = (
                elapsed / context.completed_runs
                * (context.total_runs - context.completed_runs)
            )
        else:
            eta = float(config.timeout_seconds * context.total_runs)
        self._report(context, f"Estimated time remaining: {format_eta(eta)}")

    def _report(self, context: RunContext | None, message: str) -> None:
        colors = context.colors if context is not None and context.colors else Colors("", "", "", "", "", "")
        text = f"{colors.cyan}{message}{colors.reset}"
        if context is not None and context.report is not None:
            context.report(text)
        else:
            print(text)


def _iterations(config: BenchmarkConfig) -> int:
    """Timed iterations of one nixlbench execution (``--num_iter``)."""

    return int(config.parameters["iterations"])  # type: ignore[arg-type]


def _backend(config: BenchmarkConfig) -> object:
    return config.parameters.get("backend", DEFAULT_BACKEND)


def _environment(
    config: BenchmarkConfig,
    resources: BackendResources,
    role: str,
) -> dict[str, object]:
    """Variables to set for one side's nixlbench, besides the pod's own.

    The backend's TCP defaults come first so the profile's ``env`` can override
    them, and the network pin goes last because the profile may not set it.
    """

    backend = str(_backend(config))
    user_env = config.parameters.get("env") or {}
    threads = 1 if _parallelism(config) == "processes" else int(config.parameters.get("threads", 1))
    values: dict[str, object] = dict(DEFAULT_ENV[backend])
    if backend == "Mooncake":
        # Its TCP transport does all of its I/O on this many threads (default 1), so
        # tie it to the thread count: otherwise every thread count is capped alike.
        values[MOONCAKE_IO_THREADS] = str(threads)
        if _is_enabled(user_env.get("MC_USE_TENT")):
            # With TENT the transports come from MC_TENT_CONF, and MC_FORCE_TCP would
            # select the old `tcp` transport and switch hp_tcp off.
            values.pop("MC_FORCE_TCP", None)
    address = resources.names.get(f"{role}_address")
    for name, setting in user_env.items():
        values[name] = _fill_placeholders(name, setting, threads, address)
    if config.network is not None:
        if backend == "LIBFABRIC":
            values[NETWORK_MANAGED_ENV[backend]] = config.network.for_role(role).interface
        elif backend == "Mooncake":
            # The plugin advertises one local address, the first it finds unless told.
            if address:
                values[NETWORK_MANAGED_ENV[backend]] = address
    return values


def _is_enabled(value: object) -> bool:
    return str(value).lower() not in ("", "0", "false", "n", "no", "off", "none")


_THREADS_PLACEHOLDER = re.compile(r"\{threads(?:\*(\d+))?\}")


def _fill_placeholders(name: str, value: object, threads: int, address: str | None) -> object:
    """Replace ``{threads}``, ``{threads*N}`` and ``{address}`` in an env value.

    ``{threads}`` is the thread count of the run (1 in processes mode),
    ``{threads*N}`` is N times it, for a setting that should scale faster than
    the thread count (such as connections per peer), and ``{address}`` is this
    side's secondary-network address, so one value can describe both sides and
    every point of the sweep.
    """

    if not isinstance(value, str):
        return value
    if "{address}" in value:
        if not address:
            raise ValueError(
                f"NIXL env.{name} uses {{address}}, which needs a network section"
            )
        value = value.replace("{address}", address)
    return _THREADS_PLACEHOLDER.sub(
        lambda match: str(threads * int(match.group(1) or 1)), value
    )


def _parallelism(config: BenchmarkConfig) -> object:
    return config.parameters.get("parallelism", "threads")


def _powers_of_two(start: int, maximum: int) -> list[int]:
    """Return start, 2*start, ... up to maximum, the way nixlbench steps its sweeps."""

    values = []
    value = start
    while value <= maximum:
        values.append(value)
        value *= 2
    return values


def _format_size(value: int) -> str:
    for suffix, multiplier in (("G", 1024**3), ("M", 1024**2), ("K", 1024)):
        if value % multiplier == 0:
            return f"{value // multiplier}{suffix}"
    return f"{value}B"


def _without_gpu(manifest: str) -> str:
    """Drop the GPU request and CUDA transport so the pods run on CPU-only nodes."""

    documents = [doc for doc in yaml.safe_load_all(manifest) if doc is not None]
    for document in documents:
        if document.get("kind") != "Pod":
            continue
        for container in document.get("spec", {}).get("containers", []):
            resources = container.get("resources", {})
            for section in ("requests", "limits"):
                resources.get(section, {}).pop(GPU_RESOURCE, None)
            for variable in container.get("env", []):
                if variable.get("name") == "UCX_TLS":
                    variable["value"] = CPU_UCX_TLS
    return dump_manifest(documents)


POLL_SECONDS = 0.2
PEER_GRACE_SECONDS = 5.0


def _wait_for_all(
    processes: Mapping[str, subprocess.Popen[bytes]],
    timeout: float,
) -> dict[str, int]:
    """Wait for every process, without letting one failure hang on the others.

    nixlbench's two sides wait for each other. If one exits with an error its
    peer would block until the timeout, so after a failure the survivors get
    PEER_GRACE_SECONDS to finish and are then stopped. Raises
    subprocess.TimeoutExpired if some are still running after ``timeout``.
    """

    statuses: dict[str, int] = {}
    deadline = time.monotonic() + timeout
    failed_at: float | None = None
    while len(statuses) < len(processes):
        for name, process in processes.items():
            if name in statuses:
                continue
            try:
                statuses[name] = process.wait(timeout=POLL_SECONDS)
            except subprocess.TimeoutExpired:
                continue
        now = time.monotonic()
        if failed_at is None and any(status != 0 for status in statuses.values()):
            failed_at = now
        if len(statuses) == len(processes):
            break
        if failed_at is not None and now - failed_at >= PEER_GRACE_SECONDS:
            for name, process in processes.items():
                if name not in statuses:
                    terminate_process(process)
                    statuses[name] = process.wait()
            break
        if now > deadline:
            raise subprocess.TimeoutExpired("nixlbench", timeout)
    return {name: statuses[name] for name in processes}


# nixlbench's two sides close an ASIO connection when they finish. If the initiator
# exits first, a target that is still reading the final synchronization message
# aborts with this error although every transfer has completed.
TEARDOWN_ERRORS = ("ASIO Read head failed: End of file",)


def _expected_rows(step: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """The (block size, batch size) rows an execution of this sweep step reports."""

    start_block, end_block, start_batch, end_batch = step
    return [
        (block, batch)
        for block in _powers_of_two(start_block, end_block)
        for batch in _powers_of_two(start_batch, end_batch)
    ]


def _tolerated_target_exits(
    statuses: Mapping[str, int],
    initiator_logs: Sequence[Path],
    target_logs: Sequence[Path],
    expected_rows: Sequence[tuple[int, int]],
) -> list[dict[str, JsonValue]]:
    """Return the target failures that are only a shutdown race, or [] otherwise.

    A target failure is tolerated when its initiator exited successfully with
    every expected table row in its log, and the target's error is the known
    teardown one. Anything else, including a failing initiator, is a real
    failure and nothing is tolerated for the execution.
    """

    failed = {name: status for name, status in statuses.items() if status != 0}
    if not failed or any(not name.startswith("target") for name in failed):
        return []
    tolerated: list[dict[str, JsonValue]] = []
    for index, (initiator_log, target_log) in enumerate(zip(initiator_logs, target_logs)):
        label = f" {index + 1}" if len(initiator_logs) > 1 else ""
        status = failed.get(f"target{label}")
        if status is None:
            continue
        target_text = target_log.read_text(encoding="utf-8", errors="replace")
        initiator_text = initiator_log.read_text(encoding="utf-8", errors="replace")
        if not any(error in target_text for error in TEARDOWN_ERRORS):
            return []
        if not all(_has_row(initiator_text, block, batch) for block, batch in expected_rows):
            return []
        tolerated.append({"process": f"target{label}".strip(), "status": status, "log": target_log.name})
    return tolerated


def _has_row(text: str, block_size: int, batch_size: int) -> bool:
    return (
        re.search(rf"^\s*{block_size}\s+{batch_size}\s+[0-9]+(?:\.[0-9]+)?(?:\s|$)", text, re.MULTILINE)
        is not None
    )


def _timed_bytes(
    runs: int,
    large_blk_iter_ftr: int,
    threads: int,
    block_size: int,
    batch_size: int,
) -> int:
    """Bytes one nixlbench execution moves in its timed phase for a single sweep row.

    Mirrors nixlbench: num_iter is rounded up to a multiple of threads times
    large_blk_iter_ftr and, for blocks above 1 MiB, divided by the factor.
    """

    iterations = _round_up(runs, threads * large_blk_iter_ftr)
    if block_size > LARGE_BLOCK_SIZE:
        iterations //= large_blk_iter_ftr
    return iterations * block_size * batch_size


class _RowWatcher:
    """Notes when each initiator log first shows each of its result rows.

    A row is printed when its timed phase has finished, so together with the
    bytes moved it gives when the timed phase started and ended.
    """

    INTERVAL = 0.02

    def __init__(
        self,
        logs: Sequence[Path],
        rows: Sequence[Sequence[tuple[int, int]]],
        origin: float,
    ) -> None:
        self.times: list[dict[tuple[int, int], float]] = [{} for _ in logs]
        self._logs = logs
        self._rows = rows
        self._origin = origin
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if any(self._rows):
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self._scan()  # a row printed just before the processes exited

    def _pending(self, index: int) -> list[tuple[int, int]]:
        return [row for row in self._rows[index] if row not in self.times[index]]

    def _scan(self) -> None:
        now = time.monotonic()
        for index, log in enumerate(self._logs):
            pending = self._pending(index)
            if not pending:
                continue
            try:
                text = log.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for row in pending:
                if _has_row(text, *row):
                    self.times[index][row] = now - self._origin

    def _run(self) -> None:
        while not self._stop.is_set():
            self._scan()
            if not any(self._pending(index) for index in range(len(self._logs))):
                return
            self._stop.wait(self.INTERVAL)


# Lines that report a failure: nixlbench's own errors, glog errors (E<date> ...),
# NIXL API failures and an uncaught C++ exception.
ERROR_LINE = re.compile(r"(Error|E\d{4} \d|NIXL: .*failed|terminate called|what\(\):)")


def _first_error_line(*paths: Path) -> str:
    """Return the first line a log reports as an error, if any."""

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if ERROR_LINE.match(line.lstrip()):
                return line.strip()[:300]
    return ""


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"NIXL {name} must be a mapping")
    return value


def _validate_args(value: object, backend: str = DEFAULT_BACKEND) -> None:
    for name, setting in _mapping(value, "args").items():
        if name in MANAGED_ARGS:
            raise ValueError(f"NIXL args.{name} is managed by the profile: {MANAGED_ARGS[name]}")
        if name in UNSUPPORTED_ARGS.get(backend, ()):
            raise ValueError(f"NIXL args.{name} is not supported by the {backend} backend")
        required = REQUIRED_ARGS.get(backend, {})
        if name in required and setting != required[name]:
            raise ValueError(f"NIXL args.{name} must be {str(required[name]).lower()} for the {backend} backend")
        if name in INT_ARGS:
            minimum = INT_ARGS[name]
            if isinstance(setting, bool) or not isinstance(setting, int) or setting < minimum:
                raise ValueError(f"NIXL args.{name} must be an integer >= {minimum}")
        elif name in BOOL_ARGS:
            if not isinstance(setting, bool):
                raise ValueError(f"NIXL args.{name} must be true or false")
        elif name in CHOICE_ARGS:
            if setting not in CHOICE_ARGS[name]:
                raise ValueError(
                    f"NIXL args.{name} must be one of: {', '.join(CHOICE_ARGS[name])}"
                )
        else:
            supported = sorted((*INT_ARGS, *BOOL_ARGS, *CHOICE_ARGS))
            raise ValueError(
                f"NIXL args.{name} is not supported (supported: {', '.join(supported)})"
            )


def _validate_env(
    value: object,
    *,
    network_set: bool,
    backend: str = DEFAULT_BACKEND,
) -> None:
    prefix = ENV_PREFIX.get(backend, "UCX")
    name_pattern = re.compile(rf"({prefix}|NIXL|OMP)_[A-Z0-9_]+")
    for name, setting in _mapping(value, "env").items():
        if name_pattern.fullmatch(str(name)) is None:
            raise ValueError(
                f"NIXL env.{name} must be a {prefix}_, NIXL_ or OMP_ variable name in capitals"
            )
        if name == NETWORK_MANAGED_ENV.get(backend) and network_set:
            raise ValueError(
                f"NIXL env.{name} is managed by the network section (network.*.interface)"
            )
        if isinstance(setting, bool):
            continue
        if not isinstance(setting, (str, int, float)) or "\n" in str(setting) or "\0" in str(setting):
            raise ValueError(f"NIXL env.{name} must be a string, number or boolean")


def _round_up(value: int, partition: int) -> int:
    """Round up the way nixlbench does so work divides evenly across threads."""

    return value if value % partition == 0 else value + partition - value % partition


def _check_pipeline_depth(
    *,
    depth: int,
    iterations: int,
    warmup_iterations: int,
    thread_values: Sequence[int],
    max_block_size: int,
    large_blk_iter_ftr: int = DEFAULT_LARGE_BLK_ITER_FTR,
) -> None:
    """Reject a pipeline depth that nixlbench would fail on.

    nixlbench splits the iterations across threads (and divides them by
    ``large_blk_iter_ftr`` for blocks above 1 MiB). If a thread runs fewer
    iterations than ``pipeline_depth`` it caps the depth and then aborts because
    the descriptors no longer divide evenly. Both the timed and the warmup
    iterations go through this, so the smallest per-thread count must be at
    least the depth.
    """

    if depth <= 1:
        return
    has_large_blocks = max_block_size > LARGE_BLOCK_SIZE
    for threads in thread_values:
        partition = threads * large_blk_iter_ftr
        timed = _round_up(iterations, partition)
        warmup = _round_up(warmup_iterations, partition)
        counts = {"timed iterations": timed // threads}
        if warmup > 0:
            counts["warmup iterations"] = warmup // threads
        if has_large_blocks:
            counts["timed iterations for blocks above 1 MiB"] = timed // threads // large_blk_iter_ftr
            if warmup > 0:
                counts["warmup iterations for blocks above 1 MiB"] = (
                    warmup // threads // large_blk_iter_ftr
                )
        what, limit = min(counts.items(), key=lambda item: item[1])
        if depth > limit:
            needed = depth * threads * (large_blk_iter_ftr if has_large_blocks else 1)
            raise ValueError(
                f"NIXL args.pipeline_depth {depth} exceeds the {limit} {what} that each of "
                f"{threads} thread(s) runs (iterations={iterations}, "
                f"warmup_iterations={warmup_iterations}), and nixlbench would abort. Lower "
                f"pipeline_depth to at most {limit}, or raise iterations and "
                f"warmup_iterations to at least {needed}"
            )


def _extra_flags(args: Mapping[str, object] | None) -> list[str]:
    """Return ``--name=value`` flags (the form gflags accepts for every type)."""

    flags = []
    for name, setting in (args or {}).items():
        text = ("true" if setting else "false") if isinstance(setting, bool) else str(setting)
        flags.append(f"--{name}={text}")
    return flags


def _env_prefix(env: Mapping[str, object], *, warn_unused: bool) -> list[str]:
    """Return the ``env NAME=value ...`` command prefix, or nothing for no variables."""

    assignments = _env_assignments(env, warn_unused=warn_unused)
    return ["env", *assignments] if assignments else []


def _env_assignments(
    env: Mapping[str, object] | None,
    *,
    warn_unused: bool = True,
) -> list[str]:
    """Return ``NAME=value`` items for ``env``; empty when nothing is configured.

    UCX is told to report unknown variables so that a typo cannot silently
    leave a setting unapplied.
    """

    if not env:
        return []
    values = {WARN_UNUSED: "y", **env} if warn_unused else dict(env)
    return [
        f"{name}={('y' if setting else 'n') if isinstance(setting, bool) else setting}"
        for name, setting in values.items()
    ]


def _unknown_variables(*log_texts: str) -> list[str]:
    """Return UCX variables that UCX reported as unused in the given logs."""

    names: list[str] = []
    for text in log_texts:
        for name in UNUSED_UCX_VARIABLE.findall(text):
            if name not in names:
                names.append(name)
    return names


def _minimum_total_buffer_size(
    thread_count: int,
    max_block_size: int,
    max_batch_size: int,
) -> int:
    return max_block_size * max_batch_size * thread_count
