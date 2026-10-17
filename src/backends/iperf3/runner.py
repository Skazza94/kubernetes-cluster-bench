from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

from ...benchmark.artifacts import write_json
from ...benchmark.contracts import (
    BackendResources,
    BenchmarkConfig,
    BenchmarkRunner,
    RunContext,
)
from ...benchmark.cpu import CpuSampler
from ...benchmark.kubernetes import KubectlClient, wait_for_tcp_listener
from ...benchmark.network import (
    apply_secondary_network,
    discover_addresses,
    verify_attachments,
)
from ...benchmark.process import terminate_process
from ...benchmark.resources import apply_resources
from ...benchmark.utils import (
    format_eta,
    is_nonnegative_int,
    is_positive_int,
    resource_overrides,
    utc_timestamp,
)
from ...common.console import Colors
from ...common.types import JsonValue


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "iperf3-template.yaml"
MAX_CLIENT_ATTEMPTS = 3
PROGRESS_WIDTH = 30

JsonObject = Mapping[str, object]


class Iperf3Runner(BenchmarkRunner):
    """Execute iperf3 while preserving the legacy runner behavior."""

    name = "iperf3"

    def __init__(
        self,
        manifest_template: Path = DEFAULT_MANIFEST,
        kubectl: KubectlClient | None = None,
    ) -> None:
        self.manifest_template = manifest_template
        self.kubectl = kubectl or KubectlClient()
        self._server_processes: dict[str, subprocess.Popen[bytes]] = {}
        self._colors: Colors | None = None

    def validate(self, config: BenchmarkConfig) -> None:
        parameters = config.parameters
        threads = parameters.get("threads", 1)
        if not is_positive_int(threads):
            raise ValueError("iperf3 threads must be a positive integer")

        duration = parameters.get("duration_seconds", 10)
        if not is_positive_int(duration):
            raise ValueError("duration_seconds must be a positive integer")
        warmup = parameters.get("warmup_seconds", 5)
        if not is_nonnegative_int(warmup):
            raise ValueError("warmup_seconds must be a non-negative integer")

        if not self.manifest_template.is_file():
            raise ValueError(f"iperf3 manifest not found: {self.manifest_template}")

    def render_manifest(self, config: BenchmarkConfig, destination: Path) -> None:
        self.validate(config)
        manifest = self.manifest_template.read_text(encoding="utf-8")
        replacements = {
            "__TEST_ID__": config.test_id,
            "__CLIENT_NODE__": config.initiator_node,
            "__SERVER_NODE__": config.target_node,
            "__IMAGE__": config.image,
            "__IMAGE_PULL_POLICY__": config.image_pull_policy,
        }
        missing = [marker for marker in replacements if marker not in manifest]
        if missing:
            raise ValueError(
                "iperf3 manifest is missing required placeholder(s): "
                + ", ".join(missing)
            )
        for marker, value in replacements.items():
            manifest = manifest.replace(marker, value)
        remaining = [marker for marker in replacements if marker in manifest]
        if remaining:
            raise ValueError(
                "iperf3 manifest still contains placeholder(s): "
                + ", ".join(remaining)
            )
        if config.network is not None:
            names = self.resource_names(config)
            manifest = apply_secondary_network(
                manifest,
                config.network,
                {names["client_pod"]: "initiator", names["server_pod"]: "target"},
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
        for resource in names["collision_targets"]:
            if self.kubectl.resource_exists(resource, config.namespace):
                raise ValueError(
                    f"resource already exists: {resource}; retry with a clean namespace"
                )
        try:
            self.kubectl.apply_manifest(config.namespace, manifest)
            for pod in (names["server_pod"], names["client_pod"]):
                self.kubectl.wait_for(
                    config.namespace,
                    f"pod/{pod}",
                    "jsonpath={.status.phase}=Running",
                    config.timeout_seconds,
                )
            addresses: dict[str, str] = {}
            if config.network is not None:
                addresses = discover_addresses(
                    self.kubectl,
                    config.namespace,
                    config.network,
                    {"initiator": names["client_pod"], "target": names["server_pod"]},
                )
        except BaseException:
            self.kubectl.delete_manifest(config.namespace, manifest)
            raise
        return BackendResources(
            names={
                "server_service": names["server_service"],
                "server_pod": names["server_pod"],
                "client_pod": names["client_pod"],
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
        output_directory.mkdir(parents=True, exist_ok=True)
        thread_count = int(config.parameters.get("threads", 1))
        run_number = int(config.parameters.get("run_number", 1))
        stem = f"{config.test_id}-t{thread_count}-run{run_number}"
        client_log = output_directory / f"{stem}-client.json"
        command_log = output_directory / f"{stem}-command.json"
        server_log = output_directory.parent / f"{config.test_id}-server.json"
        client_command = self.client_command(config, resources)
        server_command = self.server_command(config, resources)
        command_metadata: dict[str, JsonValue] = {
            "backend": self.name,
            "test_id": config.test_id,
            "run_number": run_number,
            "kube_context": config.kube_context,
            "initiator_node": config.initiator_node,
            "target_node": config.target_node,
            "namespace": config.namespace,
            "image": config.image,
            "parameters": dict(config.parameters),
            "network": self._network_metadata(config, resources),
            "resource_overrides": resource_overrides(config),
            "client_command": client_command,
            "server_command": server_command,
            "client_file": client_log.name,
            "server_file": f"../{server_log.name}",
            "started_at": utc_timestamp(),
        }
        write_json(command_log, command_metadata)

        server_process = self._server_processes.get(config.test_id)
        if server_process is None or server_process.poll() is not None:
            self._stop_server(config.test_id)
            server_process = self.kubectl.start_exec(
                config.namespace,
                resources.names["server_pod"],
                ["iperf3", "-s", "-J"],
                server_log,
            )
            self._server_processes[config.test_id] = server_process
            wait_for_tcp_listener(
                self.kubectl,
                config.namespace,
                resources.names["server_pod"],
                5201,
                server_process,
                float(config.timeout_seconds),
            )
            self._report(context, "Server benchmark running")

        self._report(
            context,
            "Client benchmark running with "
            f"{thread_count} thread(s) "
            f"(run {run_number}/{config.runs})",
        )
        # The client pod sends (initiator) and the server pod receives (target).
        cpu_files = {role: f"{stem}-cpu-{role}.log" for role in ("initiator", "target")}
        thread_files = {role: f"{stem}-threads-{role}.log" for role in ("initiator", "target")}
        origin = time.monotonic()
        sampler = CpuSampler(
            self.kubectl,
            config.namespace,
            {"initiator": resources.names["client_pod"], "target": resources.names["server_pod"]},
            {role: output_directory / name for role, name in cpu_files.items()},
            origin,
            thread_files={role: output_directory / name for role, name in thread_files.items()},
        )
        client_status = 1
        with sampler:
            for attempt in range(1, MAX_CLIENT_ATTEMPTS + 1):
                client_log.write_bytes(b"")
                client_process = self.kubectl.start_exec(
                    config.namespace,
                    resources.names["client_pod"],
                    self._container_command(client_command),
                    client_log,
                )
                client_status = self._wait_for_client(client_process, config, context)
                client_finished = time.monotonic() - origin
                if client_status == 0 or attempt == MAX_CLIENT_ATTEMPTS:
                    break
                colors = self._colors or Colors("", "", "", "", "", "")
                print(
                    f"{colors.yellow}benchmark attempt failed with status "
                    f"{client_status}; retrying{colors.reset}",
                    file=sys.stderr,
                )

        command_metadata["cpu_files"] = dict(cpu_files)
        command_metadata["thread_files"] = dict(thread_files)
        # The measured part of the run is its last duration_seconds, which ends
        # when the client exits.
        command_metadata["client_finished_s"] = client_finished
        command_metadata["client_status"] = client_status
        command_metadata["finished_at"] = utc_timestamp()
        write_json(command_log, command_metadata)
        return output_directory

    def cleanup(
        self,
        config: BenchmarkConfig,
        resources: BackendResources | None,
    ) -> None:
        colors = self._colors or Colors("", "", "", "", "", "")
        active = (
            config.test_id in self._server_processes
            and self._server_processes[config.test_id].poll() is None
        )
        if active:
            print(f"{colors.cyan}Stopping server benchmark session{colors.reset}")
        self._stop_server(config.test_id)
        if active:
            print(f"{colors.cyan}Server benchmark session stopped{colors.reset}")
        if resources is None or config.keep_resources or resources.manifest_path is None:
            if config.keep_resources and resources is not None:
                print(f"{colors.yellow}Keeping benchmark resources (--keep-resources){colors.reset}")
            return
        print(f"{colors.cyan}Cleaning up benchmark resources{colors.reset}")
        started = time.monotonic()
        deleted = self.kubectl.delete_manifest(config.namespace, resources.manifest_path)
        elapsed = int(time.monotonic() - started)
        if deleted:
            print(
                f"{colors.green}Benchmark resources deleted ({elapsed}s){colors.reset}"
            )

    def client_command(
        self,
        config: BenchmarkConfig,
        resources: BackendResources,
    ) -> list[str]:
        thread_count = int(config.parameters.get("threads", 1))
        command = [
            "kubectl",
            "exec",
            "-n",
            config.namespace,
            resources.names["client_pod"],
            "--",
            "iperf3",
            "-c",
            resources.names.get("target_address", resources.names["server_service"]),
            "-J",
            "-P",
            str(thread_count),
            "--get-server-output",
        ]
        if "initiator_address" in resources.names:
            command.extend(["-B", resources.names["initiator_address"]])
        duration = int(config.parameters.get("duration_seconds", 10))
        warmup = int(config.parameters.get("warmup_seconds", 5))
        command.extend(["-t", str(duration + warmup), "-O", str(warmup)])
        return command

    def server_command(
        self,
        config: BenchmarkConfig,
        resources: BackendResources,
    ) -> list[str]:
        return [
            "kubectl",
            "exec",
            "-n",
            config.namespace,
            resources.names["server_pod"],
            "--",
            "iperf3",
            "-s",
            "-J",
        ]

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

    def resource_names(self, config: BenchmarkConfig) -> dict[str, object]:
        server_name = f"network-benchmark-server-{config.test_id}"
        return {
            "server_service": server_name,
            "server_pod": server_name,
            "client_pod": f"network-benchmark-client-{config.test_id}",
            "collision_targets": (
                f"service/{server_name}",
                f"pod/{server_name}",
                f"pod/network-benchmark-client-{config.test_id}",
            ),
        }

    def _container_command(self, kubectl_command: list[str]) -> list[str]:
        marker = kubectl_command.index("--")
        return kubectl_command[marker + 1 :]

    def _wait_for_client(
        self,
        process: subprocess.Popen[bytes],
        config: BenchmarkConfig,
        context: RunContext | None,
    ) -> int:
        if context is None or not sys.stdout.isatty():
            status = process.wait()
            return status if status >= 0 else 128 + (-status)

        duration = int(config.parameters.get("duration_seconds", 10))
        warmup = int(config.parameters.get("warmup_seconds", 5))
        total_duration = duration + warmup
        elapsed_total = time.monotonic() - context.benchmark_started_monotonic
        eta_seconds = (
            elapsed_total / context.completed_runs
            * (context.total_runs - context.completed_runs)
            if context.completed_runs > 0
            else total_duration * context.total_runs
        )
        self._report(context, f"Estimated time remaining: {format_eta(eta_seconds)}")
        started = time.monotonic()
        while process.poll() is None:
            elapsed = time.monotonic() - started
            fraction = min(elapsed / total_duration, 1.0)
            _print_progress(fraction * 100, context.colors)
            try:
                # Returns as soon as the client exits, which ends the CPU window.
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        status = process.wait()
        if status == 0:
            _print_progress(100.0, context.colors)
        print()
        return status if status >= 0 else 128 + (-status)

    def _report(self, context: RunContext | None, message: str) -> None:
        colors = (
            context.colors
            if context is not None and context.colors is not None
            else Colors("", "", "", "", "", "")
        )
        formatted = f"{colors.cyan}{message}{colors.reset}"
        if context is not None and context.report is not None:
            context.report(formatted)
        else:
            print(formatted)

    def _stop_server(self, test_id: str) -> None:
        terminate_process(self._server_processes.pop(test_id, None))


def _print_progress(percent: float, colors: Colors | None = None) -> None:
    bounded = min(max(percent, 0.0), 100.0)
    filled = int(bounded * PROGRESS_WIDTH / 100)
    selected = colors or Colors("", "", "", "", "", "")
    print(
        f"\r{selected.cyan}Current run progress: "
        f"[{'#' * filled}{'.' * (PROGRESS_WIDTH - filled)}] "
        f"{bounded:3.0f}%{selected.reset}",
        end="",
        flush=True,
    )
