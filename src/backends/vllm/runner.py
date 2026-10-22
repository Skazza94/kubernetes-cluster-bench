from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Mapping

import yaml

from ...benchmark.artifacts import write_json
from ...benchmark.contracts import (
    BackendResources,
    BenchmarkConfig,
    BenchmarkRunner,
    RunContext,
)
from ...benchmark.cpu import ROLES as CPU_ROLES, CpuSampler
from ...benchmark.kubernetes import KubectlClient, KubectlError
from ...benchmark.manifest import dump_manifest
from ...benchmark.network import apply_secondary_network, discover_addresses, verify_attachments
from ...benchmark.process import terminate_process
from ...benchmark.resources import apply_resources
from ...benchmark.utils import resource_overrides, utc_timestamp
from ...common.console import Colors
from ...common.types import JsonValue
from .parser import verify


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "vllm-template.yaml"
POD_HELPER = Path(__file__).resolve().parent / "pod_helper.py"
# vLLM's NIXL-aware toy proxy, from the simulator checkout (vllm-sim/vllm_sim.py checkout).
PROXY_SOURCE = (PROJECT_ROOT / "vllm-sim" / "vllm" / "tests" / "v1" / "kv_connector"
                / "nixl_integration" / "toy_proxy_server.py")

PYTHON = "/opt/venv/bin/python"
MODEL_DIRECTORY = "/model"
MODEL_NAME = "sim-pd"
SERVER_PORT = 8000
PROXY_PORT = 8080
SIDE_CHANNEL_PORT = 5600
STEP_LOG = "/tmp/sim-steps.jsonl"
MAX_PROMPTS = 100  # distinct prompt token IDs in the fixture's vocabulary
MAX_OUTPUT_TOKENS = 32
CPU_BLOCK_MULTIPLE = 32  # vLLM's CPU attention needs block sizes divisible by 32

# execution.vllm defaults: the small functional configuration of the simulator.
DEFAULTS: dict[str, JsonValue] = {
    "prompt_lengths": "64,96",
    "output_tokens": 4,
    "block_size": 32,
    "kv_cache_memory_bytes": 8 * 1024 * 1024,
    "max_model_len": 256,
    "max_num_batched_tokens": 128,
    "max_num_seqs": 4,
    "dtype": "float32",
    "timing": "linear",
    # simulated: modeled transfers (kv_transfer_timing); real: native NIXL transfers.
    "kv_transfer_mode": "simulated",
    "startup_timeout_seconds": 600,
    "request_timeout_seconds": 120,
}
DEFAULT_TIMING_ARGS = {"fixed_us": 100, "prefill_us_per_token": 1}
DEFAULT_KV_TRANSFER_TIMING = {"load_gbps": 1, "load_latency_us": 50000}
INT_OPTIONS = (
    "output_tokens", "block_size", "kv_cache_memory_bytes", "max_model_len",
    "max_num_batched_tokens", "max_num_seqs", "startup_timeout_seconds",
    "request_timeout_seconds",
)


class VllmRunner(BenchmarkRunner):
    """vLLM prefill on the target node, decode on the initiator node, KV handoff via NIXL.

    The pods idle until their secondary-network addresses are known, then the
    runner starts ``vllm serve`` in both and the NIXL toy proxy beside prefill,
    and keeps them up for every run. Each run sends the configured prompts
    through the proxy, ``threads`` at a time, and checks from both servers'
    step logs that decode resumed from prefill's KV instead of recomputing.
    """

    name = "vllm"

    def __init__(
        self,
        manifest_template: Path = DEFAULT_MANIFEST,
        kubectl: KubectlClient | None = None,
        proxy_source: Path = PROXY_SOURCE,
    ) -> None:
        self.manifest_template = manifest_template
        self.kubectl = kubectl or KubectlClient()
        self.proxy_source = proxy_source
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._colors: Colors | None = None

    def validate(self, config: BenchmarkConfig) -> None:
        if not self.manifest_template.is_file():
            raise ValueError(f"vLLM manifest not found: {self.manifest_template}")
        if not self.proxy_source.is_file():
            raise ValueError(
                f"vLLM proxy not found: {self.proxy_source}; "
                "run `python3 vllm-sim/vllm_sim.py checkout` first"
            )
        options = _options(config)
        for name in INT_OPTIONS:
            value = options[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"vLLM {name} must be a positive integer")
        if int(options["block_size"]) % CPU_BLOCK_MULTIPLE:  # type: ignore[arg-type]
            raise ValueError(
                f"vLLM block_size must be a multiple of {CPU_BLOCK_MULTIPLE} (CPU attention)"
            )
        if int(options["output_tokens"]) > MAX_OUTPUT_TOKENS:  # type: ignore[arg-type]
            raise ValueError(f"vLLM output_tokens must be at most {MAX_OUTPUT_TOKENS}")
        lengths = _prompt_lengths(options)
        longest = min(
            int(options["max_model_len"]) - int(options["output_tokens"]),  # type: ignore[arg-type]
            int(options["max_num_batched_tokens"]),  # type: ignore[arg-type]
        )
        for length in lengths:
            if not 2 <= length <= longest:
                raise ValueError(
                    f"vLLM prompt length {length} must be between 2 and {longest} "
                    "(max_model_len - output_tokens, and at most max_num_batched_tokens "
                    "so the prompt is prefilled in one step)"
                )
        for name in ("timing_args", "kv_transfer_timing", "env"):
            value = config.parameters.get(name)
            if value is not None and not isinstance(value, dict):
                raise ValueError(f"vLLM {name} must be a mapping")
        if options["kv_transfer_mode"] not in ("simulated", "real"):
            raise ValueError("vLLM kv_transfer_mode must be simulated or real")
        if options["kv_transfer_mode"] == "real" and "kv_transfer_timing" in config.parameters:
            raise ValueError(
                "vLLM kv_transfer_timing models simulated transfers; real transfers "
                "take their actual time, so remove it"
            )

    def resource_names(self, config: BenchmarkConfig) -> dict[str, str]:
        test_id = config.test_id
        names = {
            "tools": f"vllm-pd-tools-{test_id}",
            "target_pod": f"vllm-pd-prefill-{test_id}",
            "initiator_pod": f"vllm-pd-decode-{test_id}",
        }
        names["collision_targets"] = ",".join(
            [f"configmap/{names['tools']}", f"pod/{names['target_pod']}",
             f"pod/{names['initiator_pod']}"]
        )
        return names

    def render_manifest(self, config: BenchmarkConfig, destination: Path) -> None:
        self.validate(config)
        manifest = self.manifest_template.read_text(encoding="utf-8")
        replacements = {
            "__TEST_ID__": config.test_id,
            "__IMAGE__": config.image,
            "__IMAGE_PULL_POLICY__": config.image_pull_policy,
            "__DECODE_NODE__": config.initiator_node,
            "__PREFILL_NODE__": config.target_node,
        }
        missing = [marker for marker in replacements if marker not in manifest]
        if missing:
            raise ValueError("vLLM manifest is missing placeholder(s): " + ", ".join(missing))
        for marker, value in replacements.items():
            manifest = manifest.replace(marker, value)
        names = self.resource_names(config)
        documents = [doc for doc in yaml.safe_load_all(manifest) if doc is not None]
        for document in documents:
            if document.get("kind") == "ConfigMap" and document["metadata"]["name"] == names["tools"]:
                document["data"] = {
                    "pod_helper.py": POD_HELPER.read_text(encoding="utf-8"),
                    "proxy.py": self.proxy_source.read_text(encoding="utf-8"),
                }
                break
        else:
            raise ValueError(f"vLLM manifest has no ConfigMap named {names['tools']}")
        manifest = dump_manifest(documents)
        if config.network is not None:
            manifest = apply_secondary_network(
                manifest, config.network,
                {names["initiator_pod"]: "initiator", names["target_pod"]: "target"},
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
                raise ValueError(f"resource already exists: {resource}; retry with a clean namespace")
        pods = {"initiator": names["initiator_pod"], "target": names["target_pod"]}
        logs = manifest.parent / "servers"
        try:
            self.kubectl.apply_manifest(config.namespace, manifest)
            for pod in pods.values():
                self.kubectl.wait_for(config.namespace, f"pod/{pod}",
                                      "jsonpath={.status.phase}=Running", config.timeout_seconds)
            if config.network is not None:
                # NIXL's side channel and the KV transfers must use the secondary network.
                addresses = discover_addresses(self.kubectl, config.namespace, config.network, pods)
            else:
                addresses = {
                    f"{role}_address": self.kubectl.output(
                        ["get", "pod", pod, "-n", config.namespace, "-o", "jsonpath={.status.podIP}"]
                    )
                    for role, pod in pods.items()
                }
            addresses = {key: _bare_address(value) for key, value in addresses.items()}
            options = _options(config)
            for pod in pods.values():
                self._exec(config, pod, [PYTHON, "/bench/pod_helper.py", "prepare",
                                         "--directory", MODEL_DIRECTORY,
                                         "--max-model-len", str(options["max_model_len"])])
            for role, pod in pods.items():
                command = self.server_command(config, addresses[f"{role}_address"], role)
                self._processes[role] = self.kubectl.start_exec(
                    config.namespace, pod, command, logs / f"{_side(role)}-server.log")
            self._wait_ready(config, pods, f"http://127.0.0.1:{SERVER_PORT}/health", logs)
            self._processes["proxy"] = self.kubectl.start_exec(
                config.namespace, pods["target"],
                self.proxy_command(addresses["initiator_address"]), logs / "proxy.log")
            self._wait_ready(config, {"target": pods["target"]},
                             f"http://127.0.0.1:{PROXY_PORT}/healthcheck", logs)
        except BaseException:
            self._stop_processes()
            self.kubectl.delete_manifest(config.namespace, manifest)
            raise
        return BackendResources(names={**pods, **addresses}, manifest_path=manifest)

    def server_command(self, config: BenchmarkConfig, address: str, role: str) -> list[str]:
        """``vllm serve`` for one side, with NIXL's side channel on ``address``."""

        options = _options(config)
        real = options["kv_transfer_mode"] == "real"
        simulation: dict[str, JsonValue] = {
            "clock": "real",
            "timing": options["timing"],
            "timing_args": dict(config.parameters.get("timing_args") or DEFAULT_TIMING_ARGS),  # type: ignore[arg-type]
            "log_path": STEP_LOG,
        }
        if real:
            # Only set when needed: simulator images without the option reject it.
            simulation["kv_transfer_mode"] = "real"
        else:
            simulation["kv_transfer_timing"] = dict(
                config.parameters.get("kv_transfer_timing") or DEFAULT_KV_TRANSFER_TIMING  # type: ignore[arg-type]
            )
        kv_transfer: dict[str, JsonValue] = {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_consumer" if role == "initiator" else "kv_producer",
        }
        if real:
            # Real transfers start with CPU KV buffers (DRAM registrations).
            kv_transfer["kv_buffer_device"] = "cpu"
        env = {
            # The upstream CPU image's settings for simulation without telemetry.
            "TRITON_CPU_BACKEND": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "VLLM_NIXL_SIDE_CHANNEL_HOST": address,
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(SIDE_CHANNEL_PORT),
            "OMP_NUM_THREADS": "1",
            **{name: str(value) for name, value in (config.parameters.get("env") or {}).items()},  # type: ignore[union-attr]
        }
        return [
            "env", *(f"{name}={value}" for name, value in env.items()),
            "vllm", "serve", MODEL_DIRECTORY,
            "--host", "0.0.0.0", "--port", str(SERVER_PORT),
            "--served-model-name", MODEL_NAME,
            "--dtype", str(options["dtype"]),
            "--block-size", str(options["block_size"]),
            "--kv-cache-memory-bytes", str(options["kv_cache_memory_bytes"]),
            "--max-model-len", str(options["max_model_len"]),
            "--max-num-batched-tokens", str(options["max_num_batched_tokens"]),
            "--max-num-seqs", str(options["max_num_seqs"]),
            "--generation-config", "vllm",
            "--enforce-eager",
            # Repeated runs send the same prompts; a prefix-cache hit on prefill
            # would hide the transfer being measured and fail the check.
            "--no-enable-prefix-caching",
            "--simulation-config", json.dumps(simulation),
            "--kv-transfer-config", json.dumps(kv_transfer),
            "--simulate-forward",
        ]

    def proxy_command(self, decode_address: str) -> list[str]:
        return [
            PYTHON, "/bench/proxy.py", "--host", "0.0.0.0", "--port", str(PROXY_PORT),
            "--prefiller-hosts", "127.0.0.1", "--prefiller-ports", str(SERVER_PORT),
            "--decoder-hosts", decode_address, "--decoder-ports", str(SERVER_PORT),
        ]

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
        options = _options(config)
        threads = int(config.parameters.get("threads", 1))  # type: ignore[arg-type]
        run_number = int(config.parameters.get("run_number", 1))  # type: ignore[arg-type]
        stem = f"{config.test_id}-t{threads}-run{run_number}"
        pods = {"initiator": resources.names["initiator"], "target": resources.names["target"]}
        lengths = _prompt_lengths(options)
        files = {
            "responses": f"{stem}-responses.json",
            "prefill_steps": f"{stem}-prefill-steps.jsonl",
            "decode_steps": f"{stem}-decode-steps.jsonl",
        }
        cpu_files = {role: f"{stem}-cpu-{role}.log" for role in CPU_ROLES}
        thread_files = {role: f"{stem}-threads-{role}.log" for role in CPU_ROLES}
        metadata: dict[str, JsonValue] = {
            "backend": self.name,
            "test_id": config.test_id,
            "run_number": run_number,
            "kube_context": config.kube_context,
            "initiator_node": config.initiator_node,
            "target_node": config.target_node,
            "namespace": config.namespace,
            "image": config.image,
            "parameters": dict(config.parameters),
            "options": dict(options),
            "addresses": {key: value for key, value in resources.names.items() if key.endswith("_address")},
            "server_commands": {
                role: self.server_command(config, resources.names[f"{role}_address"], role)
                for role in pods
            },
            "resource_overrides": resource_overrides(config),
            "files": files,
            "cpu_files": cpu_files,
            "thread_files": thread_files,
            "started_at": utc_timestamp(),
        }
        command_log = output_directory / f"{stem}-command.json"
        write_json(command_log, metadata)
        offsets = {role: self._step_lines(config, pod) for role, pod in pods.items()}
        self._report(context, f"Sending {len(lengths)} request(s), {threads} at a time")
        origin = time.monotonic()
        sampler = CpuSampler(
            self.kubectl, config.namespace, pods,
            {role: output_directory / name for role, name in cpu_files.items()}, origin,
            thread_files={role: output_directory / name for role, name in thread_files.items()},
        )
        timeout = int(options["request_timeout_seconds"])  # type: ignore[arg-type]
        with sampler:
            metadata["requests_started_s"] = time.monotonic() - origin
            output = self._exec(config, pods["target"], [
                PYTHON, "/bench/pod_helper.py", "requests",
                "--url", f"http://127.0.0.1:{PROXY_PORT}/v1/completions",
                "--model", MODEL_NAME,
                "--prompt-lengths", ",".join(str(length) for length in lengths),
                "--output-tokens", str(options["output_tokens"]),
                "--concurrency", str(threads),
                "--timeout", str(timeout),
            ], timeout=timeout * len(lengths) + 60)
            metadata["requests_finished_s"] = time.monotonic() - origin
        (output_directory / files["responses"]).write_text(output, encoding="utf-8")
        records = json.loads(output)["requests"]
        # Step logs are flushed shortly after the responses; retry briefly.
        error: ValueError | None = None
        for _ in range(5):
            steps = {
                role: self._exec(config, pod, ["tail", "-n", f"+{offsets[role] + 1}", STEP_LOG])
                for role, pod in pods.items()
            }
            (output_directory / files["prefill_steps"]).write_text(steps["target"], encoding="utf-8")
            (output_directory / files["decode_steps"]).write_text(steps["initiator"], encoding="utf-8")
            try:
                metadata["checks"] = verify(records, steps["target"], steps["initiator"])  # type: ignore[assignment]
                error = None
                break
            except ValueError as failure:
                error = failure
                time.sleep(1)
        metadata["status"] = "failed" if error else "passed"
        if error:
            metadata["error"] = str(error)
        metadata["finished_at"] = utc_timestamp()
        write_json(command_log, metadata)
        if error:
            raise RuntimeError(f"vLLM P/D check failed: {error}")
        self._report(context, f"P/D check passed: decode resumed from prefill's KV for "
                              f"{len(records)} request(s)")
        return output_directory

    def cleanup(self, config: BenchmarkConfig, resources: BackendResources | None) -> None:
        self._stop_processes()
        if resources is None or resources.manifest_path is None:
            return
        if config.keep_resources:
            print("Keeping benchmark resources (--keep-resources); vLLM servers were stopped")
            return
        self.kubectl.delete_manifest(config.namespace, resources.manifest_path)

    def _exec(self, config: BenchmarkConfig, pod: str, command: list[str],
              timeout: float | None = None) -> str:
        return self.kubectl.run(
            ["exec", "-n", config.namespace, pod, "--", *command],
            capture_output=True, timeout=timeout,
        ).stdout

    def _step_lines(self, config: BenchmarkConfig, pod: str) -> int:
        output = self._exec(config, pod, [
            "sh", "-c", f"if [ -f {STEP_LOG} ]; then wc -l < {STEP_LOG}; else echo 0; fi"])
        return int(output.strip() or 0)

    def _wait_ready(self, config: BenchmarkConfig, pods: Mapping[str, str], url: str,
                    logs: Path) -> None:
        """Wait until ``url`` answers in every pod, failing fast if a process exits."""

        deadline = time.monotonic() + int(_options(config)["startup_timeout_seconds"])  # type: ignore[arg-type]
        pending = dict(pods)
        while pending:
            for name, process in self._processes.items():
                if process.poll() is not None:
                    log = logs / (f"{_side(name)}-server.log" if name in CPU_ROLES else "proxy.log")
                    raise RuntimeError(f"vLLM {name} process exited early; see {log}: {_tail(log)}")
            for role, pod in list(pending.items()):
                try:
                    self._exec(config, pod, [PYTHON, "/bench/pod_helper.py", "wait",
                                             "--url", url, "--timeout", "2"], timeout=30)
                except (KubectlError, subprocess.TimeoutExpired):
                    continue
                del pending[role]
            if pending and time.monotonic() > deadline:
                raise RuntimeError(f"timed out waiting for {url} in {', '.join(pending.values())}")
            if pending:
                time.sleep(1)

    def _stop_processes(self) -> None:
        for process in self._processes.values():
            terminate_process(process)
        self._processes.clear()

    def _report(self, context: RunContext | None, message: str) -> None:
        colors = context.colors if context is not None and context.colors is not None else (
            self._colors or Colors("", "", "", "", "", ""))
        text = f"{colors.cyan}{message}{colors.reset}"
        if context is not None and context.report is not None:
            context.report(text)
        else:
            print(text)


def _options(config: BenchmarkConfig) -> dict[str, JsonValue]:
    return {name: config.parameters.get(name, default) for name, default in DEFAULTS.items()}


def _prompt_lengths(options: Mapping[str, JsonValue]) -> list[int]:
    text = str(options["prompt_lengths"])
    try:
        lengths = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise ValueError(f"vLLM prompt_lengths must be comma-separated integers, not {text!r}") from error
    if len(set(lengths)) != len(lengths):
        # The step logs identify requests by prompt length.
        raise ValueError("vLLM prompt_lengths must be distinct")
    if len(lengths) > MAX_PROMPTS:
        raise ValueError(f"vLLM prompt_lengths can list at most {MAX_PROMPTS} prompts")
    return lengths


def _bare_address(value: str) -> str:
    return value.split("/", 1)[0]


def _side(role: str) -> str:
    return "decode" if role == "initiator" else "prefill"


def _tail(path: Path, lines: int = 5) -> str:
    try:
        return " | ".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "no log"

