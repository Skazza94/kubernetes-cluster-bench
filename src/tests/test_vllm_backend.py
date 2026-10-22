from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

from src.backends.registry import default_registry
from src.backends.vllm.analyzer import VllmAnalyzer
from src.backends.vllm.parser import VllmParser, verify
from src.backends.vllm.runner import VllmRunner
from src.benchmark.contracts import BackendResources, BenchmarkConfig
from src.benchmark.network import NetworkAttachment, SecondaryNetwork
from src.graphing.cli import main as graph_main
from src.graphing.contracts import AnalysisRequest


def step_log(*requests: tuple[str, int, int, int]) -> str:
    """One simulator step line per (req_id, prompt tokens, computed, scheduled)."""

    return "".join(
        json.dumps({"step": {"requests": [{
            "req_id": req_id, "num_prompt_tokens": prompt,
            "num_computed_tokens": computed, "num_scheduled_tokens": scheduled,
        }]}}) + "\n"
        for req_id, prompt, computed, scheduled in requests
    )


def record(length: int, elapsed: float = 0.2, tokens=(70, 71, 72, 73)) -> dict:
    return {
        "index": 0, "prompt_length": length, "expected_token_ids": [70, 71, 72, 73],
        "started_s": 0.0, "finished_s": elapsed, "elapsed_s": elapsed,
        "response": {"usage": {"prompt_tokens": length},
                     "choices": [{"token_ids": list(tokens)}]},
    }


GOOD_PREFILL = step_log(("p64", 64, 0, 64), ("p96", 96, 0, 96), ("p64", 64, 64, 1))
GOOD_DECODE = step_log(("d64", 64, 63, 1), ("d96", 96, 95, 1), ("d64", 64, 64, 1))


class FakeKubectl:
    """Answers the runner's kubectl exec calls like the pods would."""

    def __init__(self, decode_steps: str = GOOD_DECODE) -> None:
        self.decode_steps = decode_steps
        self.calls: list[list[str]] = []

    def run(self, arguments, *, capture_output=False, suppress_stdout=False, timeout=None):
        self.calls.append(list(arguments))
        command = list(arguments[arguments.index("--") + 1:])
        pod = arguments[arguments.index("--") - 1]
        if command[0] == "sh":
            stdout = "0\n"
        elif command[:2] == ["/opt/venv/bin/python", "/bench/pod_helper.py"] and command[2] == "requests":
            lengths = command[command.index("--prompt-lengths") + 1].split(",")
            stdout = json.dumps({"concurrency": 1, "requests": [record(int(n)) for n in lengths]})
        elif command[0] == "tail":
            stdout = self.decode_steps if "decode" in pod else GOOD_PREFILL
        else:
            stdout = ""
        return subprocess.CompletedProcess(arguments, 0, stdout, "")


class VllmBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.proxy = self.directory / "proxy.py"
        self.proxy.write_text("# toy proxy\n", encoding="utf-8")

    def config(self, **parameters) -> BenchmarkConfig:
        return BenchmarkConfig(
            backend="vllm", test_id="t1", initiator_node="decode-node", target_node="prefill-node",
            namespace="bench", image="example/vllm-sim:cpu", log_directory=self.directory,
            network=SecondaryNetwork(NetworkAttachment("nad-d"), NetworkAttachment("nad-p")),
            parameters={"threads": 1, "run_number": 1, "prompt_lengths": "64,96", **parameters},
        )

    def runner(self, kubectl=None) -> VllmRunner:
        return VllmRunner(kubectl=kubectl or FakeKubectl(), proxy_source=self.proxy)

    def test_plugin_is_registered(self) -> None:
        plugin = default_registry().get("vllm")
        self.assertEqual(plugin.primary_metric, "latency")
        self.assertIn("prompt_length", plugin.graph_dimensions)

    def test_manifest_places_roles_and_ships_the_helper_and_proxy(self) -> None:
        destination = self.directory / "manifest.yaml"
        self.runner().render_manifest(self.config(), destination)
        documents = {doc["metadata"]["name"]: doc for doc in yaml.safe_load_all(destination.read_text())}
        tools = documents["vllm-pd-tools-t1"]
        self.assertEqual(set(tools["data"]), {"pod_helper.py", "proxy.py"})
        self.assertEqual(tools["data"]["proxy.py"], "# toy proxy\n")
        prefill, decode = documents["vllm-pd-prefill-t1"], documents["vllm-pd-decode-t1"]
        self.assertEqual(prefill["spec"]["nodeName"], "prefill-node")
        self.assertEqual(decode["spec"]["nodeName"], "decode-node")
        # Only the attachment and interface: the NAD's IPAM owns the address.
        self.assertEqual(json.loads(prefill["metadata"]["annotations"]["k8s.v1.cni.cncf.io/networks"]),
                         [{"name": "nad-p", "interface": "net1"}])
        self.assertEqual(json.loads(decode["metadata"]["annotations"]["k8s.v1.cni.cncf.io/networks"]),
                         [{"name": "nad-d", "interface": "net1"}])

    def test_invalid_options_fail_before_any_resource(self) -> None:
        runner = self.runner()
        for parameters, message in (
            ({"block_size": 48}, "multiple of 32"),
            ({"prompt_lengths": "64,64"}, "distinct"),
            ({"prompt_lengths": "200"}, "between 2 and 128"),
            ({"output_tokens": 0}, "positive integer"),
        ):
            with self.subTest(parameters=parameters), self.assertRaisesRegex(ValueError, message):
                runner.validate(self.config(**parameters))
        with self.assertRaisesRegex(ValueError, "vllm_sim.py checkout"):
            VllmRunner(proxy_source=self.directory / "missing.py").validate(self.config())

    def test_server_commands_pin_the_side_channel_and_roles(self) -> None:
        runner = self.runner()
        config = self.config(env={"MC_USE_TENT": 1})
        decode = runner.server_command(config, "10.0.0.2", "initiator")
        prefill = runner.server_command(config, "10.0.0.1", "target")
        self.assertIn("VLLM_NIXL_SIDE_CHANNEL_HOST=10.0.0.2", decode)
        self.assertIn("MC_USE_TENT=1", decode)
        self.assertIn("--no-enable-prefix-caching", decode)
        roles = [json.loads(command[command.index("--kv-transfer-config") + 1])["kv_role"]
                 for command in (decode, prefill)]
        self.assertEqual(roles, ["kv_consumer", "kv_producer"])
        self.assertEqual(runner.proxy_command("10.0.0.2")[-4:],
                         ["--decoder-hosts", "10.0.0.2", "--decoder-ports", "8000"])

    def test_real_transfer_mode_drops_the_modeled_transfer(self) -> None:
        runner = self.runner()
        command = runner.server_command(self.config(kv_transfer_mode="real"), "10.0.0.2", "initiator")
        simulation = json.loads(command[command.index("--simulation-config") + 1])
        kv_transfer = json.loads(command[command.index("--kv-transfer-config") + 1])
        self.assertEqual(simulation["kv_transfer_mode"], "real")
        self.assertNotIn("kv_transfer_timing", simulation)
        self.assertEqual(kv_transfer["kv_buffer_device"], "cpu")
        simulated = runner.server_command(self.config(), "10.0.0.2", "initiator")
        simulated_config = json.loads(simulated[simulated.index("--simulation-config") + 1])
        self.assertIn("kv_transfer_timing", simulated_config)
        self.assertNotIn("kv_transfer_mode", simulated_config)
        with self.assertRaisesRegex(ValueError, "remove it"):
            runner.validate(self.config(kv_transfer_mode="real", kv_transfer_timing={"load_gbps": 1}))
        with self.assertRaisesRegex(ValueError, "simulated or real"):
            runner.validate(self.config(kv_transfer_mode="fast"))

    def test_verify_detects_a_decode_fallback(self) -> None:
        checks = verify([record(64), record(96)], GOOD_PREFILL, GOOD_DECODE)
        self.assertEqual([check["decode_cached_tokens"] for check in checks], [63, 95])
        fallback = step_log(("d64", 64, 0, 64), ("d96", 96, 95, 1))
        with self.assertRaisesRegex(ValueError, "did not resume from prefill"):
            verify([record(64), record(96)], GOOD_PREFILL, fallback)
        with self.assertRaisesRegex(ValueError, "returned"):
            verify([record(64, tokens=(1, 2, 3, 4))], GOOD_PREFILL, GOOD_DECODE)
        with self.assertRaisesRegex(ValueError, "never ran on decode"):
            verify([record(64)], GOOD_PREFILL, "")

    def execute(self, kubectl: FakeKubectl) -> Path:
        output = self.directory / "raw" / "t1"
        resources = BackendResources(names={
            "initiator": "vllm-pd-decode-t1", "target": "vllm-pd-prefill-t1",
            "initiator_address": "10.0.0.2", "target_address": "10.0.0.1",
        })
        return self.runner(kubectl).execute(self.config(), resources, output)

    def test_execute_records_and_checks_a_run(self) -> None:
        run = self.execute(FakeKubectl())
        command = json.loads((run / "t1-t1-run1-command.json").read_text())
        self.assertEqual(command["status"], "passed")
        self.assertEqual(command["addresses"], {"initiator_address": "10.0.0.2",
                                                "target_address": "10.0.0.1"})
        self.assertLessEqual(command["requests_started_s"], command["requests_finished_s"])
        observations = VllmParser().parse(run)
        self.assertEqual([(o.metric, o.dimensions["prompt_length"], o.value) for o in observations],
                         [("latency", 64, 0.2), ("latency", 96, 0.2)])

    def test_execute_fails_when_decode_recomputes(self) -> None:
        kubectl = FakeKubectl(decode_steps=step_log(("d64", 64, 0, 64), ("d96", 96, 95, 1)))
        with self.assertRaisesRegex(RuntimeError, "did not resume"):
            self.execute(kubectl)
        command = json.loads((self.directory / "raw" / "t1" / "t1-t1-run1-command.json").read_text())
        self.assertEqual(command["status"], "failed")
        with self.assertRaisesRegex(ValueError, "did not pass"):
            VllmParser().parse(self.directory / "raw" / "t1")

    def test_latency_and_cpu_plots(self) -> None:
        run = self.execute(FakeKubectl())
        for role, cores in (("initiator", 1.0), ("target", 2.0)):
            (run / f"t1-t1-run1-cpu-{role}.log").write_text(
                "".join(f"{t - 1:.4f} {int(t * cores * 1e6)} 0 0 0\n" for t in range(0, 30)))
        command_path = run / "t1-t1-run1-command.json"
        command = json.loads(command_path.read_text())
        command.update(requests_started_s=2.0, requests_finished_s=8.0)
        command_path.write_text(json.dumps(command))
        observations = VllmAnalyzer().parse(run)
        points = VllmAnalyzer().aggregate(
            observations, AnalysisRequest(metric="latency", x_dimension="prompt_length",
                                          filters={"threads": 1}))
        self.assertEqual([(p.x_value, round(p.y_value, 6)) for p in points], [(64, 200.0), (96, 200.0)])
        arguments = ["--backend", "vllm", "--name", self.directory.name,
                     "--dir", str(self.directory.parent),
                     "--x", "threads", "--filter", "prompt_length=64"]
        self.assertEqual(graph_main([*arguments, "--y", "latency"]), 0)
        self.assertEqual(graph_main([*arguments, "--y", "cpu"]), 0)
        self.assertEqual(graph_main([*arguments, "--y", "cpu-per-gb"]), 2)


if __name__ == "__main__":
    unittest.main()
