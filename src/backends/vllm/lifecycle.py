"""Deploy, run and collect evidence for vLLM dataset workloads."""
from __future__ import annotations

import base64
import ipaddress
import json
from pathlib import Path
import shutil
import time
import uuid

from ...benchmark.artifacts import write_json
from ...benchmark.contracts import BackendResources
from ...benchmark.cpu import CpuSampler, cleanup_problems
from ...benchmark.network import discover_addresses, verify_attachments
from ...benchmark.process import terminate_process
from . import accounting, config as settings, inputs

PYTHON = "/opt/venv/bin/python"


def pod_identity(runner, config, pod):
    value = json.loads(runner.kubectl.output(["get", "pod", pod, "-n", config.namespace, "-o", "json"]))
    statuses = value["status"]["containerStatuses"]
    if len(statuses) != 1 or not statuses[0]["ready"]:
        raise ValueError(f"{pod}: expected one ready simulator container")
    status = statuses[0]
    expected = config.image.split("@", 1)[1]
    image_id = status["imageID"]
    if image_id.split("@")[-1].removeprefix("docker-pullable://") != expected:
        raise ValueError(f"{pod}: runtime image identity {image_id} does not match {config.image}")
    pod_ip = str(ipaddress.IPv4Address(value["status"]["podIP"]))
    return {"uid": value["metadata"]["uid"], "node": value["spec"]["nodeName"],
            "pod": pod, "namespace": config.namespace,
            "image_requested": config.image, "image_id": image_id, "container_id": status["containerID"], "pod_ip": pod_ip}


def capture(runner, config, pod, role, path):
    o = settings.options(config)
    tent = o["connector"] == "mooncake" or o["nixl_backend"] == "mooncake"
    output = runner._exec(config, pod, [PYTHON, "/bench/evidence.py", "snapshot",
                          str(o[f"{role}_data_parallel_size"]), str(o[f"{role}_tensor_parallel_size"]),
                          str(tent).lower()], timeout=120)
    path.write_text(output)
    return json.loads(output)


def deploy(runner, config, manifest):
    o = settings.options(config)
    names = runner.resource_names(config)
    pods = {"initiator": names["initiator_pod"], "target": names["target_pod"]}
    logs = manifest.parent / "servers"
    logs.mkdir(exist_ok=False)
    receipt = {"status": "deploying", "image_requested": config.image,
               "options": o, "source": inputs.source_receipt(config, runner._proxy_source(config))}
    write_json(logs / "deployment.json", receipt)
    applied = False
    try:
        if config.network is not None:
            verify_attachments(runner.kubectl, config.namespace, config.network)
        for resource in names["collision_targets"].split(","):
            if runner.kubectl.resource_exists(resource, config.namespace):
                raise ValueError(f"resource already exists: {resource}")
        applied = True
        runner.kubectl.apply_manifest(config.namespace, manifest)
        for pod in pods.values():
            runner.kubectl.wait_for(config.namespace, f"pod/{pod}", "jsonpath={.status.phase}=Running", config.timeout_seconds)
        addresses = (discover_addresses(runner.kubectl, config.namespace, config.network, pods)
                     if config.network is not None else {
                         f"{role}_address": runner.kubectl.output(["get", "pod", pod, "-n", config.namespace,
                                                                  "-o", "jsonpath={.status.podIP}"])
                         for role, pod in pods.items()})
        addresses = {key: value.split("/", 1)[0] for key, value in addresses.items()}
        receipt["pods"] = {role: pod_identity(runner, config, pod) for role, pod in pods.items()}
        for role, node in (("target", config.target_node), ("initiator", config.initiator_node)):
            if receipt["pods"][role]["node"] != node:
                raise ValueError(f"{role}: resolved pod is on another node")
        paths, captured = inputs.stage(config, runner.kubectl, pods, manifest.parent / "inputs")
        runner._staged_timing_path = paths.get("timing_path")
        receipt["inputs"] = captured
        receipt["server_commands"] = {role: runner.server_command(config, addresses[f"{role}_address"], role) for role in pods}
        # HTTP uses the primary pod network; KV uses the secondary interface.
        receipt["http_endpoints"] = {"prefill": "127.0.0.1:8000",
                                     "decode": receipt["pods"]["initiator"]["pod_ip"] + ":8000",
                                     "client_proxy": "127.0.0.1:8080"}
        native = runner.proxy_command(receipt["pods"]["initiator"]["pod_ip"], config, "127.0.0.1")
        receipt["proxy_command"] = [PYTHON, "/bench/ledger.py", "proxy", "/tmp/kcb-proxy-ledger.jsonl", *native[1:]]
        write_json(logs / "deployment.json", receipt)
        for role, pod in pods.items():
            side = "prefill" if role == "target" else "decode"
            runner._processes[role] = runner.kubectl.start_exec(config.namespace, pod, receipt["server_commands"][role], logs / f"{side}-server.log")
        runner._wait_ready(config, pods, "http://127.0.0.1:8000/health", logs)
        runner._processes["proxy"] = runner.kubectl.start_exec(config.namespace, pods["target"], receipt["proxy_command"], logs / "proxy.log")
        runner._wait_ready(config, {"target": pods["target"]}, "http://127.0.0.1:8080/healthcheck", logs)
        for role, pod in pods.items():
            side = "prefill" if role == "target" else "decode"
            snapshot = capture(runner, config, pod, side, logs / f"{side}-startup.json")
            for dp in range(o[f"{side}_data_parallel_size"]):
                for tp in range(o[f"{side}_tensor_parallel_size"]):
                    raw = accounting.require(snapshot["ranks"][f"dp{dp}.tp{tp}"]["steps"], "rank startup")
                    rows = accounting.records(raw)
                    # Native thread records are emitted lazily on the first
                    # step. MoE DP0 alone emits the joint timing record.
                    for key in (("kv_layout", "timing") if dp == 0 else ("kv_layout",)):
                        if not any(key in r for r in rows):
                            raise ValueError(f"{side}/dp{dp}.tp{tp}: missing startup {key}")
            if not snapshot["server_processes"]:
                raise ValueError(f"{side}: missing server process epoch")
        receipt["status"] = "ready"
        write_json(logs / "deployment.json", receipt)
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(logs / "deployment.json", receipt)
        # Preserve partial per-rank logs before owner-scoped pod deletion.
        if applied:
            for role, pod in pods.items():
                try:
                    capture(runner, config, pod, "prefill" if role == "target" else "decode", logs / f"{role}-failure.json")
                except Exception as failure:
                    (logs / f"{role}-collection-error.txt").write_text(str(failure))
        runner._stop_processes()
        if applied:
            runner.kubectl.delete_manifest(config.namespace, manifest)
        raise
    return BackendResources(names={**pods, **addresses, **paths}, manifest_path=manifest)


def execute(runner, config, resources, output_directory, context=None):
    o = settings.options(config)
    threads, number = int(config.parameters.get("threads", 1)), int(config.parameters.get("run_number", 1))
    stem = f"{config.test_id}-t{threads}-run{number}"
    directory = output_directory / stem
    directory.mkdir(parents=True, exist_ok=False)
    command_path = output_directory / f"{stem}-command.json"
    deployment = resources.manifest_path.parent
    metadata = {"status": "collecting", "backend": "vllm", "options": o,
                "parameters": dict(config.parameters), "test_id": config.test_id, "run_number": number,
                "evidence_directory": directory.name}
    write_json(command_path, metadata)
    manifest = {"options": o, "collection_errors": [], "expected_instances": {"prefill": 1, "decode": 1},
                "interfaces": {}, "artifact_hashes": {}, "profilers_stopped": False, "samplers_stopped": False}
    manifest["expected_ranks"] = [{"role": role, "instance": 0, "dp": dp, "tp": tp}
                                  for role in ("prefill", "decode")
                                  for dp in range(o[f"{role}_data_parallel_size"])
                                  for tp in range(o[f"{role}_tensor_parallel_size"])]
    manifest["image_requested"] = config.image
    manifest["expected_nodes"] = {"prefill": config.target_node, "decode": config.initiator_node}
    pods = {"prefill": resources.names["target"], "decode": resources.names["initiator"]}
    remote = "/tmp/kcb-run-" + uuid.uuid4().hex
    processes = {}
    sampler = None
    interrupted = None
    client_attempted = False
    def preserve_error(label, error):
        manifest["collection_errors"].append(f"{label}: {type(error).__name__}: {error}")
        import re
        key = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label)
        for stream in ("stdout", "stderr", "output"):
            value = getattr(error, stream, None)
            if value:
                path = directory / f"{key}-{stream}.txt"
                path.write_bytes(value if isinstance(value, bytes) else str(value).encode())
    def collect(label, action):
        try:
            return action()
        except Exception as error:
            preserve_error(label, error)
    try:
        shutil.copytree(deployment / "inputs", directory / "inputs")
        receipt = json.loads((deployment / "servers" / "deployment.json").read_text())
        write_json(directory / "deployment.json", receipt)
        for role, pod in pods.items():
            generic = "target" if role == "prefill" else "initiator"
            current = pod_identity(runner, config, pod)
            write_json(directory / f"{role}-pod-before.json", current)
            if current != receipt["pods"][generic]:
                raise ValueError(f"{role}: pod/image/container changed after deployment")
            # Re-verify retained input hashes immediately before each repetition.
            expected = {"/model/"+name: digest for name, digest in receipt["inputs"]["model"]["sha256"].items()}
            for kind in ("timing", "dataset"):
                item = receipt["inputs"].get(kind, {})
                if item.get("sha256") and (kind != "dataset" or role == "prefill"):
                    expected[item["pod_path"]] = item["sha256"]
            actual = json.loads(runner._exec(config, pod, [PYTHON, "/bench/evidence.py", "hashes", json.dumps(expected)], timeout=300))
            write_json(directory / f"{role}-input-hashes.json", actual)
            if actual != expected:
                raise ValueError(f"{role}: staged inputs changed")
            # Discover the attachment's *configured* pod interface, without
            # changing its ownership or allocating another one.
            if config.network is None:
                raise ValueError("dataset KV accounting requires a configured secondary network attachment")
            attachment = getattr(config.network, generic)
            manifest["interfaces"][role] = attachment.interface
            before = capture(runner, config, pod, role, directory / f"{role}-before.json")
            accounting.cleanup_metrics(accounting.require(before["metrics"], f"{role} initial metrics"), o[f"{role}_data_parallel_size"])
        # Retain the previous proxy prefix; later repetitions get only their
        # segment, while the cumulative ledger remains preserved in the pod.
        proxy_before = runner._exec(config, pods["prefill"], ["sh", "-c", "cat /tmp/kcb-proxy-ledger.jsonl 2>/dev/null || true"])
        (directory / "proxy-ledger-before.jsonl").write_text(proxy_before)
        generic_pods = {"target": pods["prefill"], "initiator": pods["decode"]}
        stopped = runner._stop_profilers(config, generic_pods)
        from .runner import _profiler_problems
        if _profiler_problems(stopped):
            raise ValueError(f"prior profiler cleanup failed: {stopped}")
        duration = float(o["request_timeout_seconds"]) * int(o["num_prompts"]) + 60
        files = {f"{role}_profile": f"{role}-profile.jsonl" for role in pods}
        processes = runner._start_profilers(config, generic_pods, directory, files, duration)
        if o["profile_interval_seconds"] and set(processes) != set(generic_pods):
            raise ValueError("required existing profiler did not start for both servers")
        sampler = CpuSampler(runner.kubectl, config.namespace, generic_pods,
                             {r: directory / f"cpu-{r}.log" for r in generic_pods}, time.monotonic(),
                             thread_files={r: directory / f"threads-{r}.log" for r in generic_pods})
        command = runner.bench_command(config, resources, threads) + ["--artifact-directory", remote]
        write_json(directory / "client-command.json", command)
        with sampler:
            if not sampler.cleanups or set(sampler.cleanups[0]) != set(generic_pods) or cleanup_problems(sampler.cleanups[0]):
                raise ValueError("prior CPU sampler cleanup not confirmed on both pods")
            # The helper persists last-client.json even if its exec exits
            # nonzero. Finally collects it and every partial client artifact.
            client_attempted = True
            stdout = runner._exec(config, pods["prefill"], command, timeout=duration)
            (directory / "client-stdout.txt").write_text(stdout)
            if float(o["settle_seconds"]):
                time.sleep(float(o["settle_seconds"]))
    except BaseException as error:
        preserve_error("execution", error)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            manifest["interrupted"] = True
            interrupted = error
    finally:
        generic_pods = {"target": pods["prefill"], "initiator": pods["decode"]}
        if client_attempted:
            report = collect("stop client", lambda: json.loads(runner._exec(config, pods["prefill"], [PYTHON, "/bench/evidence.py", "stop-client", remote], timeout=30)))
            manifest["client_cleanup"] = report
            if report is None or report.get("stopped") is not True:
                preserve_error("stop client", ValueError("client process group not confirmed stopped"))
        stopped = collect("stop profilers", lambda: runner._stop_profilers(config, generic_pods))
        from .runner import _profiler_problems
        manifest["profiler_cleanup"] = stopped
        manifest["profilers_stopped"] = bool(stopped is not None and set(stopped) == set(pods) and not _profiler_problems(stopped))
        for process in processes.values():
            terminate_process(process)
        manifest["sampler_cleanup"] = sampler.cleanups if sampler is not None else []
        manifest["samplers_stopped"] = bool(sampler is not None and sampler.cleanups
                                               and set(sampler.cleanups[-1]) == set(generic_pods)
                                               and not cleanup_problems(sampler.cleanups[-1]))
        for role, pod in pods.items():
            def final_identity():
                identity = pod_identity(runner, config, pod)
                write_json(directory / f"{role}-pod-after.json", identity)
                deployed = json.loads((directory / "deployment.json").read_text())["pods"]["target" if role == "prefill" else "initiator"]
                if identity != deployed:
                    raise ValueError("pod/container/image epoch changed")
            collect(f"{role} final pod identity", final_identity)
            # Drain is a local service envelope, never labelled native latency.
            # Retain every attempt; no overwritten evidence or failed scrape zeros.
            deadline = time.monotonic() + 30
            attempt = 0
            while True:
                path = directory / f"{role}-drain-{attempt}.json"
                value = collect(f"{role} drain {attempt}", lambda: capture(runner, config, pod, role, path))
                if value is None:
                    break
                shutil.copyfile(path, directory / f"{role}-after.json")
                try:
                    accounting.cleanup_metrics(accounting.require(value["metrics"], "final metrics"), o[f"{role}_data_parallel_size"])
                    break
                except ValueError:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.2)
                    attempt += 1
            collect(f"{role} connections", lambda: (directory / f"{role}-connections.json").write_text(
                runner._exec(config, pod, [PYTHON, "/bench/pod_helper.py", "connections"], timeout=30)))
        def client_files():
            raw = runner._exec(config, pods["prefill"], [PYTHON, "/bench/evidence.py", "files", remote], timeout=120)
            (directory / "client-bundle.json").write_text(raw)
            bundle = json.loads(raw)
            target = directory / "client-raw"
            target.mkdir()
            for name, encoded in bundle.items():
                path = target / name
                if not path.resolve().is_relative_to(target.resolve()):
                    raise ValueError("unsafe client artifact path")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base64.b64decode(encoded, validate=True))
            payload = json.loads((target / "last-client.json").read_text())
            write_json(directory / "client.json", payload)
            ledgers = list(target.glob("*/client-ledger.jsonl"))
            if len(ledgers) != 1:
                raise ValueError("missing/ambiguous client ledger")
            shutil.copyfile(ledgers[0], directory / "client-ledger.jsonl")
        collect("client files", client_files)
        def proxy_files():
            full = runner._exec(config, pods["prefill"], ["cat", "/tmp/kcb-proxy-ledger.jsonl"], timeout=30)
            (directory / "proxy-ledger-full.jsonl").write_text(full)
            old = (directory / "proxy-ledger-before.jsonl").read_text()
            if not full.startswith(old):
                raise ValueError("proxy ledger epoch changed")
            (directory / "proxy-ledger.jsonl").write_text(full[len(old):])
        collect("proxy ledger", proxy_files)
        for path in (deployment / "servers").glob("*.log"):
            collect(f"server log {path.name}", lambda path=path: shutil.copyfile(path, directory / path.name))
        # Hash all captured inputs/receipts, stdout, startup, raw profiles and
        # partial evidence. The manifest/derived verdict are deliberately outside
        # their own hash inventory so summaries can be regenerated.
        manifest["artifact_hashes"] = {str(path.relative_to(directory)): inputs.sha256(path)
                                      for path in directory.rglob("*") if path.is_file() and path.name not in ("manifest.json", "accounting.json")}
        write_json(directory / "manifest.json", manifest)
        verdict = accounting.regenerate(directory)
        write_json(directory / "accounting.json", verdict)
        metadata.update(status=verdict["status"], accounting=verdict)
        write_json(command_path, metadata)
    if interrupted is not None:
        raise interrupted
    if verdict["status"] != "passed":
        raise ValueError(f"vLLM evidence failed; retained at {directory}: {verdict['problems'][:5]}")
    return command_path


def stop_servers(runner, config, resources):
    reports = {}
    for role in ("target", "initiator"):
        try:
            reports[role] = json.loads(runner._exec(config, resources.names[role], [PYTHON, "/bench/evidence.py", "stop-servers"], timeout=30))
        except Exception as error:
            reports[role] = {"stopped": False, "error": str(error)}
    if resources.manifest_path:
        write_json(resources.manifest_path.parent / "servers" / "cleanup.json", reports)
    if config.keep_resources and not all(r.get("stopped") for r in reports.values()):
        raise ValueError(f"server shutdown not confirmed; cleanup reports: {reports}")
    return reports


def cleanup(runner, config, resources):
    try:
        reports = stop_servers(runner, config, resources)
    finally:
        runner._stop_processes()
        runner._staged_timing_path = None
    if config.keep_resources:
        print("Keeping benchmark resources (--keep-resources); server/proxy shutdown confirmed")
        return
    deleted = runner.kubectl.delete_manifest(config.namespace, resources.manifest_path)
    write_json(resources.manifest_path.parent / "servers" / "resource-cleanup.json",
               {"servers": reports, "manifest_deleted": deleted})
    if not deleted:
        raise ValueError("owner-scoped simulator resource deletion failed; see resource-cleanup.json")
