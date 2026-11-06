from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

from src.backends.plugin import BackendPlugin
from src.backends.registry import BackendRegistry
from src.benchmark.cli import parse_config
from src.benchmark.contracts import BackendResources, BenchmarkConfig, RunContext
from src.benchmark.cli import main


class CliRunner:
    def validate(self, config: BenchmarkConfig) -> None:
        return None

    def render_manifest(self, config: BenchmarkConfig, destination: Path) -> None:
        destination.write_text("kind: Pod\n", encoding="utf-8")

    def deploy(self, config: BenchmarkConfig, manifest: Path) -> BackendResources:
        return BackendResources(manifest_path=manifest)

    def execute(
        self,
        config: BenchmarkConfig,
        resources: BackendResources,
        output_directory: Path,
        context: RunContext | None = None,
    ) -> Path:
        output_directory.joinpath("result.txt").write_text("ok\n", encoding="utf-8")
        return output_directory

    def cleanup(
        self,
        config: BenchmarkConfig,
        resources: BackendResources | None,
    ) -> None:
        return None


class BenchmarkCliTests(unittest.TestCase):
    def test_only_declared_timing_arguments_accept_nested_json(self) -> None:
        profile = {
            "backend": "vllm", "nodes": {"initiator": "decode-node", "target": "prefill-node"},
            "execution": {"image": "example/sim@sha256:" + "a" * 64, "namespace": "bench",
                          "vllm": {"timing": "aisimulate", "timing_args": {
                              "gpu": "h100", "quant_modes": {"weight": "bf16"}}}},
        }
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested.yaml"
            def parse():
                path.write_text(yaml.safe_dump(profile))
                with patch("src.benchmark.cli.KubectlClient.context_and_namespace", return_value=("offline", "bench")):
                    return parse_config(["--profile", str(path)])
            config = parse()
            self.assertEqual(config.parameters["timing_args"]["quant_modes"], {"weight": "bf16"})
            for key in ("env", "bench_args", "hp_tcp_args"):
                profile["execution"]["vllm"][key] = {"unexpected": {"nested": True}}
                with self.assertRaisesRegex(ValueError, "scalar"):
                    parse()
                del profile["execution"]["vllm"][key]
            profile["execution"]["vllm"]["timing_args"] = {"host_us": float("nan")}
            with self.assertRaisesRegex(ValueError, "finite JSON"):
                parse()
            profile["execution"]["vllm"]["timing_args"] = {}
            profile["execution"]["vllm"]["pipeline_parallel_size"] = 2
            with self.assertRaisesRegex(ValueError, "pipeline_parallel_size"):
                parse()

    def test_parse_config_reads_common_and_backend_parameters(self) -> None:
        profile = """\
backend: fake
name: friendly-name
test_id: test-1
nodes:
  initiator: worker-a
  target: worker-b
dimensions:
  threads: 1:4:2
execution:
  image: example/fake:latest
  runs: 3
  fake:
    runtime: ASIO
    num_iter: 1000
"""
        registry = BackendRegistry(
            (
                BackendPlugin(
                    name="fake",
                    profile_options=frozenset({"runtime", "num_iter"}),
                    profile_dimensions=frozenset({"threads"}),
                ),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.yaml"
            profile_path.write_text(profile, encoding="utf-8")
            config = parse_config(["--profile", str(profile_path)], registry)
        self.assertEqual(config.parameters["threads"], 1)
        self.assertEqual(config.parameters["threads_spec"], "1:4:2")
        self.assertEqual(config.parameters["runtime"], "ASIO")
        self.assertEqual(config.parameters["num_iter"], 1000)
        self.assertEqual(config.runs, 3)
        self.assertEqual(config.name, "friendly-name")

    def test_threads_can_be_an_exact_list(self) -> None:
        registry = BackendRegistry(
            (BackendPlugin(name="fake", profile_dimensions=frozenset({"threads"})),)
        )
        for value, expected in (('"1,2,4,6,8"', "1,2,4,6,8"), ("[1, 2, 4, 8]", "1,2,4,8"), ("4", "4")):
            profile = (
                "backend: fake\nnodes:\n  initiator: a\n  target: b\n"
                f"dimensions:\n  threads: {value}\nexecution:\n  image: x\n"
            )
            with tempfile.TemporaryDirectory() as directory:
                profile_path = Path(directory) / "profile.yaml"
                profile_path.write_text(profile, encoding="utf-8")
                config = parse_config(["--profile", str(profile_path)], registry)
            self.assertEqual(config.parameters["threads_spec"], expected)
        for bad in ("[1, true]", "[1, 2.5]", "[1, a]"):
            profile = (
                "backend: fake\nnodes:\n  initiator: a\n  target: b\n"
                f"dimensions:\n  threads: {bad}\nexecution:\n  image: x\n"
            )
            with tempfile.TemporaryDirectory() as directory:
                profile_path = Path(directory) / "profile.yaml"
                profile_path.write_text(profile, encoding="utf-8")
                with self.assertRaises(ValueError, msg=bad):
                    parse_config(["--profile", str(profile_path)], registry)

    def test_profile_uses_declared_image(self) -> None:
        profile = """\
backend: nixl
nodes:
  initiator: worker-a
  target: worker-b
dimensions:
  threads: 1
  block_size: "4M:64M"
  batch_size: "1:8"
execution:
  namespace: bench
  image: example/nixl:latest
  nixl:
    iterations: 100
    warmup_iterations: 10
"""
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "nixl.yaml"
            profile_path.write_text(profile, encoding="utf-8")
            config = parse_config(["--profile", str(profile_path)])
        self.assertEqual(config.image, "example/nixl:latest")
        self.assertEqual(config.parameters["block_size_start"], 4194304)
        self.assertEqual(config.parameters["block_size_max"], 67108864)
        self.assertEqual(config.parameters["batch_size_start"], 1)
        self.assertEqual(config.parameters["batch_size_max"], 8)

    def test_nixl_dimensions_require_power_of_two_ranges(self) -> None:
        profile = """\
backend: nixl
nodes:
  initiator: worker-a
  target: worker-b
dimensions:
  threads: 1
  block_size: "3M:64M"
  batch_size: "1:8"
execution:
  image: example/nixl:latest
  nixl:
    iterations: 100
    warmup_iterations: 10
"""
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "invalid-nixl.yaml"
            profile_path.write_text(profile, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "powers of two"):
                parse_config(["--profile", str(profile_path)])

    def test_max_consecutive_failures_profile_field(self) -> None:
        template = """\
backend: iperf3
nodes:
  initiator: worker-a
  target: worker-b
execution:
  image: example/image:latest
  namespace: bench
%s"""
        def load(extra: str):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "p.yaml"
                path.write_text(template % extra, encoding="utf-8")
                return parse_config(["--profile", str(path)])

        self.assertEqual(load("").max_consecutive_failures, 5)
        self.assertEqual(load("  max_consecutive_failures: 0\n").max_consecutive_failures, 0)
        self.assertEqual(load("  max_consecutive_failures: 3\n").max_consecutive_failures, 3)
        with self.assertRaises(ValueError):
            load("  max_consecutive_failures: -1\n")

    def test_profile_templates_load(self) -> None:
        profiles = Path(__file__).resolve().parents[2] / "profiles"
        for backend in ("iperf3", "nixl"):
            config = parse_config(
                ["--profile", str(profiles / f"{backend}-template.yaml")]
            )
            self.assertEqual(config.backend, backend)
            self.assertNotEqual(config.initiator_node, config.target_node)

    def test_image_pull_policy_profile_field(self) -> None:
        template = """\
backend: iperf3
nodes:
  initiator: worker-a
  target: worker-b
execution:
  image: example/image:latest
  namespace: bench
%s"""
        def load(extra: str):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "p.yaml"
                path.write_text(template % extra, encoding="utf-8")
                return parse_config(["--profile", str(path)])

        self.assertEqual(load("").image_pull_policy, "Always")
        self.assertEqual(load("  image_pull_policy: Never\n").image_pull_policy, "Never")
        with self.assertRaisesRegex(ValueError, "image_pull_policy"):
            load("  image_pull_policy: Sometimes\n")

    def test_network_profile_section(self) -> None:
        template = """\
backend: %s
nodes:
  initiator: worker-a
  target: worker-b
execution:
  image: example/image:latest
  namespace: bench
%s
%s"""
        backend_sections = {
            "iperf3": "",
            "nixl": '  nixl: {iterations: 10, warmup_iterations: 1}\ndimensions: {block_size: "4M", batch_size: 1}',
        }
        def load(backend: str, extra: str):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "p.yaml"
                path.write_text(
                    template % (backend, backend_sections[backend], extra),
                    encoding="utf-8",
                )
                return parse_config(["--profile", str(path)])

        self.assertIsNone(load("iperf3", "").network)
        full = (
            "network:\n"
            "  initiator: {attachment: secondary-a}\n"
            "  target: {attachment: ns/secondary-b, interface: net2, ip: 10.0.0.2/24}\n"
        )
        network = load("iperf3", full).network
        self.assertEqual(network.initiator.attachment, "secondary-a")
        self.assertEqual(network.initiator.interface, "net1")
        self.assertIsNone(network.initiator.ip)
        self.assertEqual(network.target.attachment, "ns/secondary-b")
        self.assertEqual(network.target.interface, "net2")
        self.assertEqual(network.target.ip, "10.0.0.2/24")
        with self.assertRaisesRegex(ValueError, "network.target' is required"):
            load("iperf3", "network:\n  initiator: {attachment: secondary-a}\n")
        with self.assertRaisesRegex(ValueError, "initiator.attachment"):
            load("iperf3", "network:\n  initiator: {interface: net1}\n  target: {attachment: b}\n")
        with self.assertRaisesRegex(ValueError, "unsupported field"):
            load("iperf3", "network:\n  attachment: secondary\n  initiator: {attachment: a}\n  target: {attachment: b}\n")
        with self.assertRaisesRegex(ValueError, "network.target: ip must include"):
            load("iperf3", "network:\n  initiator: {attachment: a}\n  target: {attachment: b, ip: 10.0.0.2}\n")
        self.assertEqual(load("nixl", full).network.target.interface, "net2")

        unsupported = BackendRegistry((BackendPlugin(name="fake"),))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p.yaml"
            path.write_text(
                "backend: fake\nnodes: {initiator: a, target: b}\n"
                "execution: {image: x/y:z, namespace: bench}\n" + full,
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not support"):
                parse_config(["--profile", str(path)], unsupported)

    def test_resources_profile_section(self) -> None:
        template = """\
backend: iperf3
nodes:
  initiator: worker-a
  target: worker-b
execution:
  image: example/image:latest
  namespace: bench
%s"""
        def load(extra: str):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "p.yaml"
                path.write_text(template % extra, encoding="utf-8")
                return parse_config(["--profile", str(path)])

        self.assertTrue(load("").resources.is_empty)
        resources = load(
            "  resources:\n    requests: {cpu: 2, memory: 4Gi}\n    limits: {cpu: 4}\n"
        ).resources
        self.assertEqual(dict(resources.requests), {"cpu": "2", "memory": "4Gi"})
        self.assertEqual(dict(resources.limits), {"cpu": "4"})
        with self.assertRaisesRegex(ValueError, "unsupported field"):
            load("  resources:\n    requests: {gpu: 1}\n")
        with self.assertRaisesRegex(ValueError, "unsupported field"):
            load("  resources:\n    cpu: 2\n")
        with self.assertRaisesRegex(ValueError, "limits.memory"):
            load("  resources:\n    limits: {memory: lots}\n")
        with self.assertRaisesRegex(ValueError, "must be a string or number"):
            load("  resources:\n    limits: {cpu: true}\n")

    def test_nixl_args_and_env_profile_sections(self) -> None:
        template = """\
backend: nixl
nodes:
  initiator: worker-a
  target: worker-b
dimensions: {block_size: "4M", batch_size: 1}
execution:
  image: example/image:latest
  namespace: bench
  nixl:
    iterations: 10
    warmup_iterations: 1
%s"""
        def load(extra: str):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "p.yaml"
                path.write_text(template % extra, encoding="utf-8")
                return parse_config(["--profile", str(path)])

        config = load("    args: {pipeline_depth: 16, enable_pt: true}\n    env: {UCX_ZCOPY_THRESH: 1K, UCX_TCP_NODELAY: false}\n")
        self.assertEqual(config.parameters["args"], {"pipeline_depth": 16, "enable_pt": True})
        self.assertEqual(config.parameters["env"], {"UCX_ZCOPY_THRESH": "1K", "UCX_TCP_NODELAY": False})
        self.assertNotIn("args", load("").parameters)
        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            load("    args: [pipeline_depth]\n")
        with self.assertRaisesRegex(ValueError, "scalar values"):
            load("    env: {UCX_TLS: [tcp]}\n")

    def test_nixl_iterations_are_separate_from_runs(self) -> None:
        template = """\
backend: nixl
nodes:
  initiator: worker-a
  target: worker-b
dimensions: {block_size: "4M", batch_size: 1}
execution:
  image: example/image:latest
  namespace: bench
%s"""
        def load(execution: str):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "p.yaml"
                path.write_text(template % execution, encoding="utf-8")
                return parse_config(["--profile", str(path)])

        config = load("  runs: 3\n  nixl: {iterations: 512, warmup_iterations: 64}\n")
        self.assertEqual(config.runs, 3)
        self.assertEqual(config.parameters["iterations"], 512)
        # A profile written before the split must not turn its iterations into runs.
        with self.assertRaisesRegex(ValueError, "execution.nixl.iterations' is required"):
            load("  runs: 512\n  nixl: {warmup_iterations: 64}\n")
        with self.assertRaisesRegex(ValueError, "warmup_runs was renamed: use execution.nixl.warmup_iterations"):
            load("  nixl: {iterations: 512, warmup_runs: 64}\n")
        with self.assertRaisesRegex(ValueError, "repetitions was renamed: use execution.runs"):
            load("  nixl: {iterations: 512, warmup_iterations: 64, repetitions: 3}\n")

    def test_missing_image_is_rejected(self) -> None:
        profile = """\
backend: iperf3
nodes:
  initiator: worker-a
  target: worker-b
execution: {}
"""
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "missing-image.yaml"
            profile_path.write_text(profile, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "execution.image"):
                parse_config(["--profile", str(profile_path)])

    def test_default_name_uses_test_id(self) -> None:
        profile = """\
backend: iperf3
test_id: test-1
nodes:
  initiator: worker-a
  target: worker-b
execution:
  image: example/iperf3:latest
"""
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "iperf3.yaml"
            profile_path.write_text(profile, encoding="utf-8")
            config = parse_config(["--profile", str(profile_path)])
        self.assertEqual(config.output_name, "test-1")

    def test_parse_iperf3_duration_options(self) -> None:
        profile = """\
backend: iperf3
nodes:
  initiator: worker-a
  target: worker-b
dimensions:
  threads: 1
execution:
  image: example/iperf3:latest
  iperf3:
    duration_seconds: 5
    warmup_seconds: 2
"""
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "iperf3.yaml"
            profile_path.write_text(profile, encoding="utf-8")
            config = parse_config(["--profile", str(profile_path)])
        self.assertEqual(config.parameters["duration_seconds"], 5)
        self.assertEqual(config.parameters["warmup_seconds"], 2)

    def test_main_dispatches_through_registry(self) -> None:
        registry = BackendRegistry(
            (BackendPlugin(name="fake", runner_factory=CliRunner),)
        )
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "fake.yaml"
            profile_path.write_text(
                f"""\
backend: fake
test_id: test-1
nodes:
  initiator: worker-a
  target: worker-b
execution:
  image: example/fake:latest
  log_directory: {directory}
""",
                encoding="utf-8",
            )
            status = main(["--profile", str(profile_path)], registry)
        self.assertEqual(status, 0)

    def test_equal_nodes_are_rejected(self) -> None:
        profile = """\
backend: iperf3
nodes:
  initiator: worker-a
  target: worker-a
execution:
  image: example/iperf3:latest
"""
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "invalid.yaml"
            profile_path.write_text(profile, encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_config(["--profile", str(profile_path)])


if __name__ == "__main__":
    unittest.main()
