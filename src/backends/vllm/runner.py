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
from ...benchmark.cpu import ROLES as CPU_ROLES, CpuSampler, cleanup_problems
from ...benchmark.kubernetes import KubectlClient, KubectlError
from ...benchmark.manifest import dump_manifest
from ...benchmark.network import apply_secondary_network, discover_addresses, verify_attachments
from ...benchmark.process import terminate_process
from ...benchmark.resources import apply_resources
from ...benchmark.utils import resource_overrides, utc_timestamp
from ...common.console import Colors
from ...common.types import JsonValue
from . import config as sim_config
from . import inputs
from .parser import StepLog, check_scenario, summarize, verify
from .profile import summarize_profile, summarize_reads, summarize_tent


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "vllm-template.yaml"
POD_HELPER = Path(__file__).resolve().parent / "pod_helper.py"

PYTHON = "/opt/venv/bin/python"
MODEL_DIRECTORY = "/model"
MODEL_NAME = "sim-pd"
SERVER_PORT = 8000
PROXY_PORT = 8080
# TENT's own metrics, served in the pod only (images built with them).
TENT_METRICS_PORT = 9100
STEP_LOG = "/tmp/sim-steps.jsonl"
MAX_REQUESTS = 400
SCENARIOS = ("proxy", "handoff", "cancel", "producer-failure", "pressure")
MAX_OUTPUT_TOKENS = 32

DEFAULTS = sim_config.DEFAULTS


class VllmRunner(BenchmarkRunner):
    """Run prefill on the target node and decode on the initiator node.

    A connector-specific proxy routes requests. Collect client results,
    rank logs and transfer checks for each workload.
    """

    name = "vllm"

    def __init__(
        self,
        manifest_template: Path = DEFAULT_MANIFEST,
        kubectl: KubectlClient | None = None,
        proxy_source: Path | None = None,
    ) -> None:
        self.manifest_template = manifest_template
        self.kubectl = kubectl or KubectlClient()
        self.proxy_source = proxy_source
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._colors: Colors | None = None

    def _proxy_source(self, config: BenchmarkConfig) -> Path:
        return self.proxy_source or inputs.proxy_source(config)

    def validate(self, config: BenchmarkConfig) -> None:
        o = sim_config.validate(config)
        if not self.manifest_template.is_file():
            raise ValueError(f"vLLM manifest not found: {self.manifest_template}")
        if not self._proxy_source(config).is_file():
            raise ValueError(f"vLLM proxy not found: {self._proxy_source(config)}; check the vllm/ source directory")
        if o["workload"] == "fixture":
            if o["output_tokens"] > MAX_OUTPUT_TOKENS:
                raise ValueError(f"fixture output_tokens must be at most {MAX_OUTPUT_TOKENS}")
            if o["scenario"] not in SCENARIOS:
                raise ValueError("vLLM scenario must be one of: " + ", ".join(SCENARIOS))
            if o["scenario"] != "proxy" and o["kv_transfer_mode"] != "real":
                raise ValueError(f"vLLM scenario {o['scenario']} needs kv_transfer_mode real")
            if o["scenario"] == "producer-failure" and (config.runs != 1 or any(
                c in str(config.parameters.get("threads_spec", "1")) for c in (",", ":")
            )):
                raise ValueError("producer-failure needs one run and one thread count")
            longest = min(int(o["max_model_len"]) - int(o["output_tokens"]), int(o["max_num_batched_tokens"]))
            if any(not 2 <= length <= longest for length in _prompt_lengths(o)):
                raise ValueError(f"fixture prompt length must be between 2 and {longest}")
            _cancel_delays(o)

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
                    "proxy.py": self._proxy_source(config).read_text(encoding="utf-8"),
                    **{name: (POD_HELPER.parent / name).read_text(encoding="utf-8")
                       for name in ("ledger.py", "evidence.py")},
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
        if _options(config)["workload"] == "dataset":
            from .lifecycle import deploy
            return deploy(self, config, manifest)
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
        o = _options(config)
        # Use the same content-addressed path as input staging.
        timing_path = getattr(self, "_staged_timing_path", None)
        if timing_path is None:
            _, timing_path, _ = inputs.timing_input(config)
        sim, kv, env = sim_config.server_settings(config, address, role, timing_path)
        side = "decode" if role == "initiator" else "prefill"
        command = ["env", *(f"{k}={v}" for k, v in env.items()), "vllm", "serve", MODEL_DIRECTORY,
                   "--host", "0.0.0.0", "--port", str(SERVER_PORT), "--served-model-name", MODEL_NAME]
        for flag, value in (("dtype", o["dtype"]), ("kv-cache-dtype", o["kv_cache_dtype"]),
                            ("block-size", o["block_size"]), ("kv-cache-memory-bytes", o["kv_cache_memory_bytes"]),
                            ("max-model-len", o["max_model_len"]), ("max-num-batched-tokens", o["max_num_batched_tokens"]),
                            ("max-num-seqs", o["max_num_seqs"]), ("tensor-parallel-size", o[f"{side}_tensor_parallel_size"]),
                            ("data-parallel-size", o[f"{side}_data_parallel_size"])):
            if value is not None:
                command += [f"--{flag}", str(value)]
        for key in ("enable_expert_parallel", "language_model_only", "trust_remote_code"):
            if o[key]:
                command.append("--" + key.replace("_", "-"))
        command += ["--generation-config", "vllm", "--enforce-eager",
                    "--enable-prefix-caching" if o["enable_prefix_caching"] else "--no-enable-prefix-caching",
                    "--simulation-config", json.dumps(sim), "--kv-transfer-config", json.dumps(kv), "--simulate-forward"]
        return command

    def side_env(self, config: BenchmarkConfig, address: str) -> dict[str, str]:
        return sim_config.server_settings(config, address, "initiator")[2]

    def transport_env(self, config: BenchmarkConfig, address: str) -> dict[str, str]:
        return {k: v for k, v in self.side_env(config, address).items()
                if k.startswith(("MC_", "TENT_", "NIXL_", "UCX_", "FI_"))}

    def proxy_command(self, decode_address: str, config: BenchmarkConfig | None = None,
                      prefill_address: str = "127.0.0.1") -> list[str]:
        command = [PYTHON, "/bench/proxy.py", "--host", "0.0.0.0", "--port", str(PROXY_PORT)]
        if config is not None and _options(config)["connector"] == "mooncake":
            return command + ["--prefill", f"http://{prefill_address}:{SERVER_PORT}", "8998",
                              "--decode", f"http://{decode_address}:{SERVER_PORT}"]
        return command + ["--prefiller-hosts", prefill_address, "--prefiller-ports", str(SERVER_PORT),
                          "--decoder-hosts", decode_address, "--decoder-ports", str(SERVER_PORT)]

    def bench_command(self, config: BenchmarkConfig, resources: BackendResources, threads: int) -> list[str]:
        o = _options(config)
        path = resources.names.get("dataset_path") or o["dataset_path"]
        # Only the HF dataset client may resolve remote dataset assets. Servers
        # keep the manifest's offline environment and use staged model inputs.
        client_env = ["env", "HF_HUB_OFFLINE=0", "HF_DATASETS_OFFLINE=0"] if o["dataset_name"] == "hf" else []
        return [*client_env, PYTHON, "/bench/pod_helper.py", "bench", "--base-url", f"http://127.0.0.1:{PROXY_PORT}",
                "--model", MODEL_NAME, "--tokenizer", MODEL_DIRECTORY, "--concurrency", str(threads),
                "--num-prompts", str(o["num_prompts"]), "--dataset-name", str(o["dataset_name"]),
                *(["--dataset-path", str(path)] if path else []), "--request-rate", str(o["request_rate"]),
                "--burstiness", str(o["burstiness"]), "--seed", str(o["seed"]),
                "--bench-args", json.dumps(o.get("bench_args", {})),
                *(["--trust-remote-code"] if o["trust_remote_code"] else [])]

    def configuration_receipt(self, config: BenchmarkConfig, addresses: Mapping[str, str]) -> dict:
        self.validate(config)
        _, _, timing = inputs.timing_input(config)
        return {"status": "prepared", "image_requested": config.image,
                "runtime_image_verified": False, "source": inputs.source_receipt(config, self._proxy_source(config)),
                "options": _options(config), "timing_input": timing,
                "server_commands": {role: self.server_command(config, address, role)
                                    for role, address in addresses.items()},
                "proxy_command": self.proxy_command(addresses["initiator"], config, addresses["target"])}

    def execute(
        self,
        config: BenchmarkConfig,
        resources: BackendResources,
        output_directory: Path,
        context: RunContext | None = None,
    ) -> Path:
        self.validate(config)
        if _options(config)["workload"] == "dataset":
            from .lifecycle import execute
            return execute(self, config, resources, output_directory, context)
        self._colors = context.colors if context is not None else self._colors
        output_directory.mkdir(parents=True, exist_ok=True)
        options = _options(config)
        threads = int(config.parameters.get("threads", 1))  # type: ignore[arg-type]
        run_number = int(config.parameters.get("run_number", 1))  # type: ignore[arg-type]
        stem = f"{config.test_id}-t{threads}-run{run_number}"
        pods = {"initiator": resources.names["initiator"], "target": resources.names["target"]}
        lengths = _prompt_lengths(options) * int(options["repeat"])  # type: ignore[arg-type]
        scenario = str(options["scenario"])
        files = {
            "responses": f"{stem}-responses.json",
            "prefill_steps": f"{stem}-prefill-steps.jsonl",
            "decode_steps": f"{stem}-decode-steps.jsonl",
            "prefill_metrics": f"{stem}-prefill-metrics.txt",
            "decode_metrics": f"{stem}-decode-metrics.txt",
            "decode_connections": f"{stem}-decode-connections.json",
            "prefill_profile": f"{stem}-prefill-profile.jsonl",
            "decode_profile": f"{stem}-decode-profile.jsonl",
            "prefill_tent_before": f"{stem}-prefill-tent-before.txt",
            "decode_tent_before": f"{stem}-decode-tent-before.txt",
            "prefill_tent": f"{stem}-prefill-tent.txt",
            "decode_tent": f"{stem}-decode-tent.txt",
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
        self._fetch_tent(config, pods, output_directory, files, "_tent_before")
        self._report(context, f"Scenario {scenario}: {len(lengths)} request(s), {threads} at a time")
        origin = time.monotonic()
        sampler = CpuSampler(
            self.kubectl, config.namespace, pods,
            {role: output_directory / name for role, name in cpu_files.items()}, origin,
            thread_files={role: output_directory / name for role, name in thread_files.items()},
        )
        timeout = int(options["request_timeout_seconds"])  # type: ignore[arg-type]
        scenario_timeout = (timeout * (len(lengths) + len(_cancel_delays(options)) + 3) + 60
                            + float(options["pause_after_cancel_seconds"]))  # type: ignore[arg-type]
        # Also stops any a failed run left behind: none may be left to start.
        problems = _profiler_problems(self._stop_profilers(config, pods))
        if problems:
            raise RuntimeError(f"cannot start: profilers not confirmed stopped: {problems}")
        profilers = self._start_profilers(config, pods, output_directory, files,
                                          scenario_timeout + float(options["settle_seconds"]))  # type: ignore[arg-type]
        try:
            output = self._run_scenario(config, resources, pods, sampler, scenario, lengths,
                                        threads, timeout, scenario_timeout, options, metadata, origin)
            if float(options["settle_seconds"]):  # type: ignore[arg-type]
                self._report(context, f"Waiting {options['settle_seconds']} s before reading metrics")
                time.sleep(float(options["settle_seconds"]))  # type: ignore[arg-type]
        finally:
            stopped = self._stop_profilers(config, pods)
            for process in profilers.values():
                terminate_process(process)
        metadata["profilers_stopped"] = stopped  # type: ignore[assignment]
        (output_directory / files["responses"]).write_text(output, encoding="utf-8")
        scenario_output = json.loads(output)
        records = scenario_output["requests"]
        for key in ("decode_healthy_after", "decode_health_error"):
            if key in scenario_output:
                metadata[key] = scenario_output[key]
        # Step logs and metrics settle shortly after the responses; retry briefly.
        error: ValueError | None = None
        metadata["samplers_stopped"] = [  # type: ignore[assignment]
            {_side(role): text for role, text in cleanup.items()} for cleanup in sampler.cleanups]
        left = _profiler_problems(stopped) + cleanup_problems(sampler.cleanups[-1] if sampler.cleanups else {})
        if left:
            error = ValueError(f"CPU samplers or profilers not confirmed stopped after the run: {left}")
        for _ in range(0 if error else 5):
            self._collect(config, pods, output_directory, files)
            steps = {
                role: self._exec(config, pod, ["tail", "-n", f"+{offsets[role] + 1}", STEP_LOG])
                for role, pod in pods.items()
            }
            (output_directory / files["prefill_steps"]).write_text(steps["target"], encoding="utf-8")
            (output_directory / files["decode_steps"]).write_text(steps["initiator"], encoding="utf-8")
            try:
                metadata["checks"] = verify(  # type: ignore[assignment]
                    records, steps["target"], steps["initiator"],
                    kv_check=sim_config.effective_pattern_flags(options)["kv_pattern_check"],
                    scenario=scenario,
                    real=options["kv_transfer_mode"] == "real",
                )
                metadata["decode_connections"] = json.loads(
                    (output_directory / files["decode_connections"]).read_text() or "[]")
                metadata["summary"] = summarize(  # type: ignore[assignment]
                    metadata["checks"],  # type: ignore[arg-type]
                    (output_directory / files["prefill_metrics"]).read_text(),
                    (output_directory / files["decode_metrics"]).read_text(),
                    steps["target"], steps["initiator"],
                )
                check_scenario(scenario, metadata["summary"], metadata)  # type: ignore[arg-type]
                error = None
                break
            except ValueError as failure:
                error = failure
                time.sleep(1)
        if not error:
            self._describe(config, pods, output_directory, files, steps, metadata)
        metadata["status"] = "failed" if error else "passed"
        if error:
            metadata["error"] = str(error)
        metadata["finished_at"] = utc_timestamp()
        write_json(command_log, metadata)
        if error:
            raise RuntimeError(f"vLLM P/D check failed: {error}")
        outcomes = metadata["summary"]["outcomes"]  # type: ignore[index]
        self._report(context, f"Scenario {scenario} passed: {outcomes}")
        return output_directory

    def _run_scenario(self, config: BenchmarkConfig, resources: BackendResources,
                      pods: Mapping[str, str], sampler: CpuSampler, scenario: str,
                      lengths: list[int], threads: int, timeout: int, scenario_timeout: float,
                      options: Mapping[str, JsonValue], metadata: dict[str, JsonValue],
                      origin: float) -> str:
        """The scenario's requests, sent by the helper in the prefill pod
        while the CPU sampler runs; returns its output."""

        with sampler:
            # The sampler stopped any a failed run left in the pods first.
            problems = cleanup_problems(sampler.cleanups[0]) if sampler.cleanups else ["no cleanup"]
            if problems:
                raise RuntimeError(f"cannot start: CPU samplers not confirmed stopped: {problems}")
            metadata["requests_started_s"] = time.monotonic() - origin
            # The helper runs in the prefill pod: prefill and the proxy are local,
            # decode is reached at its secondary-network address.
            output = self._exec(config, pods["target"], [
                PYTHON, "/bench/pod_helper.py", "scenario", scenario,
                "--proxy", f"http://127.0.0.1:{PROXY_PORT}",
                "--prefill", f"http://127.0.0.1:{SERVER_PORT}",
                "--decode", f"http://{resources.names['initiator_address']}:{SERVER_PORT}",
                "--model", MODEL_NAME,
                "--prompt-lengths", ",".join(str(length) for length in lengths),
                "--output-tokens", str(options["output_tokens"]),
                "--concurrency", str(threads),
                "--timeout", str(timeout),
                "--cancel-delays", ",".join(str(delay) for delay in _cancel_delays(options)),
                "--pause-after-cancel", str(options["pause_after_cancel_seconds"]),
            ], timeout=scenario_timeout)
            metadata["requests_finished_s"] = time.monotonic() - origin
        return output

    def _stop_profilers(self, config: BenchmarkConfig, pods: Mapping[str, str]) -> dict[str, dict]:
        """Stop the profilers in both pods and report what each stopped and
        what, if anything, is still running (``remaining``). Stopping the
        local kubectl leaves the pod's process running."""

        results: dict[str, dict] = {}
        for role, pod in pods.items():
            try:
                results[_side(role)] = json.loads(self._exec(
                    config, pod, [PYTHON, "/bench/pod_helper.py", "stop-profilers"], timeout=30))
            except (KubectlError, subprocess.TimeoutExpired, ValueError) as failure:
                results[_side(role)] = {"error": str(failure)}
        return results

    def _start_profilers(self, config: BenchmarkConfig, pods: Mapping[str, str],
                         output_directory: Path, files: Mapping[str, str],
                         duration: float) -> dict[str, subprocess.Popen[bytes]]:
        """``pod_helper.py profile`` in both pods, each into its file; best
        effort, like the CPU sampler."""

        interval = float(_options(config)["profile_interval_seconds"])  # type: ignore[arg-type]
        profilers = {}
        for role, pod in pods.items() if interval else ():
            try:
                profilers[role] = self.kubectl.start_exec(
                    config.namespace, pod,
                    [PYTHON, "/bench/pod_helper.py", "profile",
                     "--interval", str(interval), "--duration", str(duration + 60)],
                    output_directory / files[f"{_side(role)}_profile"])
            except (KubectlError, OSError):
                continue
        return profilers

    def _describe(self, config: BenchmarkConfig, pods: Mapping[str, str], output_directory: Path,
                  files: Mapping[str, str], steps: Mapping[str, str],
                  metadata: dict[str, JsonValue]) -> None:
        """What the run ran on and where its CPU and network went: each
        server's registered KV layout and CPU affinity, decode's reads, and
        each profile by server state. Best effort: what cannot be read is
        left out or reported as an error."""

        for role, pod in pods.items():
            side = _side(role)
            try:
                layout = self._exec(config, pod, ["grep", "-m1", '"kv_layout"', STEP_LOG], timeout=30)
                if layout.strip():
                    metadata.setdefault("kv_layout", {})[side] = json.loads(layout)["kv_layout"]  # type: ignore[index, union-attr]
                affinity = self._exec(config, pod, [PYTHON, "/bench/pod_helper.py", "affinity"], timeout=30)
                metadata.setdefault("affinity", {})[side] = json.loads(affinity)  # type: ignore[index, union-attr]
            except (KubectlError, subprocess.TimeoutExpired, ValueError, KeyError):
                pass
            before, after = (output_directory / files[f"{side}{suffix}"] for suffix in ("_tent_before", "_tent"))
            if before.is_file() and after.is_file():
                tent = summarize_tent(before.read_text(encoding="utf-8"), after.read_text(encoding="utf-8"))
                if tent:
                    metadata.setdefault("tent", {})[side] = tent  # type: ignore[index, union-attr]
            path = output_directory / files[f"{side}_profile"]
            if path.is_file():
                try:
                    # Threads are logged once, as they start: not only in this run's lines.
                    threads = self._exec(config, pod, ["grep", '"threads"', STEP_LOG], timeout=30)
                except (KubectlError, subprocess.TimeoutExpired):
                    threads = ""
                try:
                    profile = summarize_profile(path.read_text(encoding="utf-8"),
                                                StepLog.parse(threads + steps[role]), producer=role == "target")
                except (ValueError, KeyError, TypeError) as failure:
                    profile = {"error": str(failure)}
                metadata.setdefault("profile", {})[side] = profile  # type: ignore[index, union-attr]
        reads = summarize_reads(StepLog.parse(steps["initiator"]))
        if reads is not None:
            metadata["reads"] = reads

    def _collect(self, config: BenchmarkConfig, pods: Mapping[str, str], output_directory: Path,
                 files: Mapping[str, str]) -> None:
        """Both servers' /metrics, TENT's, and decode's established TCP
        connections. Best effort: a dead server (producer-failure) leaves an
        empty file."""

        for role, name in (("target", "prefill_metrics"), ("initiator", "decode_metrics")):
            try:
                text = self._exec(config, pods[role], [
                    PYTHON, "/bench/pod_helper.py", "fetch",
                    "--url", f"http://127.0.0.1:{SERVER_PORT}/metrics"], timeout=60)
            except (KubectlError, subprocess.TimeoutExpired):
                text = ""
            (output_directory / files[name]).write_text(text, encoding="utf-8")
        self._fetch_tent(config, pods, output_directory, files, "_tent")
        try:
            connections = self._exec(config, pods["initiator"], [
                PYTHON, "/bench/pod_helper.py", "connections"], timeout=60)
        except (KubectlError, subprocess.TimeoutExpired):
            connections = "[]"
        (output_directory / files["decode_connections"]).write_text(connections, encoding="utf-8")

    def cleanup(self, config: BenchmarkConfig, resources: BackendResources | None) -> None:
        if _options(config)["workload"] == "dataset" and resources is not None and resources.manifest_path is not None:
            from .lifecycle import cleanup
            return cleanup(self, config, resources)
        self._stop_processes()
        if resources is None or resources.manifest_path is None:
            return
        if config.keep_resources:
            print("Keeping benchmark resources (--keep-resources); vLLM servers were stopped")
            return
        self.kubectl.delete_manifest(config.namespace, resources.manifest_path)

    def _fetch_tent(self, config: BenchmarkConfig, pods: Mapping[str, str], output_directory: Path,
                    files: Mapping[str, str], suffix: str) -> None:
        """TENT's metrics in each pod (cumulative; empty without them)."""

        if _options(config)["kv_transfer_mode"] != "real":
            return
        for role, pod in pods.items():
            try:
                text = self._exec(config, pod, [
                    PYTHON, "/bench/pod_helper.py", "fetch",
                    "--url", f"http://127.0.0.1:{TENT_METRICS_PORT}/metrics"], timeout=60)
            except (KubectlError, subprocess.TimeoutExpired):
                text = ""
            (output_directory / files[f"{_side(role)}{suffix}"]).write_text(text, encoding="utf-8")

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
    return sim_config.options(config)


def _prompt_lengths(options: Mapping[str, JsonValue]) -> list[int]:
    text = str(options["prompt_lengths"])
    try:
        lengths = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise ValueError(f"vLLM prompt_lengths must be comma-separated integers, not {text!r}") from error
    if len(lengths) * int(options.get("repeat", 1)) > MAX_REQUESTS:  # type: ignore[arg-type]
        raise ValueError(f"vLLM prompt_lengths times repeat can be at most {MAX_REQUESTS} requests")
    return lengths


def _cancel_delays(options: Mapping[str, JsonValue]) -> list[float]:
    if options["scenario"] != "cancel":
        return []
    try:
        delays = [float(value) for value in str(options["cancel_delays"]).split(",") if value]
    except ValueError as error:
        raise ValueError("vLLM cancel_delays must be comma-separated seconds") from error
    if not delays or any(delay <= 0 for delay in delays):
        raise ValueError("vLLM cancel_delays must list positive seconds")
    return delays


def _bare_address(value: str) -> str:
    return value.split("/", 1)[0]


def _profiler_problems(results: Mapping[str, dict]) -> list[str]:
    """What a ``_stop_profilers`` round failed to confirm: a stop that failed
    (the helper exits nonzero if any profiler survives) or one left running."""

    return [f"{side}: {result.get('error') or 'still running: ' + str(result['remaining'])}"
            for side, result in results.items() if "error" in result or result.get("remaining")]


def _side(role: str) -> str:
    return "decode" if role == "initiator" else "prefill"


def _tail(path: Path, lines: int = 5) -> str:
    try:
        return " | ".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "no log"

