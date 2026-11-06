from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.backends.iperf3.runner import Iperf3Runner
from dataclasses import replace

from src.benchmark.contracts import BackendResources, BenchmarkConfig
from src.benchmark.kubernetes import KubectlError
from src.benchmark.network import NetworkAttachment, SecondaryNetwork


class FakeProcess:
    def __init__(self, status: int = 0) -> None:
        self.status = status

    def poll(self):
        return self.status

    def wait(self, timeout=None):
        return self.status


class Iperf3RunnerTests(unittest.TestCase):
    def config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            backend="iperf3", test_id="test-1", initiator_node="worker-a",
            target_node="worker-b", namespace="bench", image="example/image:latest",
            log_directory=Path("logs"), parameters={
                "threads": 2, "duration_seconds": 5, "warmup_seconds": 1,
            },
        )

    def resources(self) -> BackendResources:
        return BackendResources(names={
            "server_service": "server", "server_pod": "server", "client_pod": "client",
        })

    def test_validation_rejects_invalid_values(self) -> None:
        runner = Iperf3Runner()
        for key, value in (("threads", 0), ("duration_seconds", 0), ("warmup_seconds", -1)):
            config = self.config()
            config = BenchmarkConfig(**{**config.__dict__, "parameters": {**config.parameters, key: value}})
            with self.assertRaises(ValueError):
                runner.validate(config)

    def test_execute_and_cleanup_with_mocked_processes(self) -> None:
        runner = Iperf3Runner(kubectl=Mock())
        def start_exec(namespace, pod, command, path):
            path.write_text("{}", encoding="utf-8")
            return FakeProcess()
        runner.kubectl.start_exec.side_effect = start_exec
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.backends.iperf3.runner.wait_for_tcp_listener"):
                output = runner.execute(self.config(), self.resources(), Path(directory) / "raw" / "t1")
            self.assertTrue(output.is_dir())
            runner.cleanup(self.config(), BackendResources(manifest_path=Path("manifest.yaml")))
        runner.kubectl.delete_manifest.assert_called_once()

    def test_client_connects_to_service_by_default(self) -> None:
        command = Iperf3Runner().client_command(self.config(), self.resources())
        self.assertEqual(command[command.index("-c") + 1], "server")
        self.assertNotIn("-B", command)

    def test_client_uses_secondary_addresses(self) -> None:
        resources = BackendResources(names={
            **self.resources().names,
            "initiator_address": "192.168.50.6", "target_address": "192.168.50.7",
        })
        command = Iperf3Runner().client_command(self.config(), resources)
        self.assertEqual(command[command.index("-c") + 1], "192.168.50.7")
        self.assertEqual(command[command.index("-B") + 1], "192.168.50.6")

    def test_deploy_discovers_each_sides_interface_and_checks_requested_ip(self) -> None:
        kubectl = Mock()
        kubectl.resource_exists.side_effect = lambda resource, namespace: (
            resource.startswith("network-attachment-definitions")
        )
        def output(args):
            if args[2].startswith("network-benchmark-client"):
                return '[{"interface": "net1", "ips": ["192.168.50.6"]}]'
            return '[{"interface": "net2", "ips": ["192.168.60.7"]}]'
        kubectl.output.side_effect = output
        runner = Iperf3Runner(kubectl=kubectl)
        network = SecondaryNetwork(
            initiator=NetworkAttachment("secondary-a", "net1"),
            target=NetworkAttachment("secondary-b", "net2", "192.168.60.7/24"),
        )
        config = replace(self.config(), network=network)
        resources = runner.deploy(config, Path("manifest.yaml"))
        self.assertEqual(resources.names["initiator_address"], "192.168.50.6")
        self.assertEqual(resources.names["target_address"], "192.168.60.7")

        bad = replace(config, network=SecondaryNetwork(
            network.initiator, NetworkAttachment("secondary-b", "net2", "192.168.60.99/24")
        ))
        with self.assertRaisesRegex(ValueError, "requested 192.168.60.99"):
            runner.deploy(bad, Path("manifest.yaml"))
        kubectl.delete_manifest.assert_called_once()

    def test_deploy_rejects_missing_attachment_or_multus(self) -> None:
        network = SecondaryNetwork(
            NetworkAttachment("secondary-a"), NetworkAttachment("ns2/secondary-b")
        )
        config = replace(self.config(), network=network)
        kubectl = Mock()
        kubectl.resource_exists.side_effect = lambda resource, namespace: (
            namespace == "bench"
        )
        with self.assertRaisesRegex(
            ValueError, r"'secondary-b' \(network.target\) not found in namespace 'ns2'"
        ):
            Iperf3Runner(kubectl=kubectl).deploy(config, Path("manifest.yaml"))
        kubectl.resource_exists.assert_any_call(
            "network-attachment-definitions.k8s.cni.cncf.io/secondary-a", "bench"
        )
        kubectl.apply_manifest.assert_not_called()

        kubectl = Mock()
        kubectl.resource_exists.side_effect = KubectlError(
            ["get"], 1,
            'error: the server doesn\'t have a resource type "network-attachment-definitions"',
        )
        with self.assertRaisesRegex(ValueError, "requires Multus"):
            Iperf3Runner(kubectl=kubectl).deploy(config, Path("manifest.yaml"))
        kubectl.apply_manifest.assert_not_called()

    def test_deploy_checks_shared_attachment_once_and_ignores_other_errors(self) -> None:
        shared = NetworkAttachment("secondary")
        config = replace(self.config(), network=SecondaryNetwork(shared, shared))
        kubectl = Mock()
        kubectl.resource_exists.side_effect = [
            KubectlError(["get"], 1, "forbidden"), False, False, False,
        ]
        kubectl.output.return_value = '[{"interface": "net1", "ips": ["10.0.0.5"]}]'
        resources = Iperf3Runner(kubectl=kubectl).deploy(config, Path("manifest.yaml"))
        self.assertEqual(resources.names["target_address"], "10.0.0.5")
        self.assertEqual(kubectl.resource_exists.call_count, 4)  # 1 NAD + 3 pods/service

    def test_resources_override_cpu_and_memory(self) -> None:
        import yaml
        from src.benchmark.resources import PodResources

        config = replace(
            self.config(),
            resources=PodResources({"cpu": "4", "memory": "4Gi"}, {"cpu": "4", "memory": "4Gi"}),
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "manifest.yaml"
            Iperf3Runner().render_manifest(config, destination)
            docs = [d for d in yaml.safe_load_all(destination.read_text(encoding="utf-8")) if d["kind"] == "Pod"]
        self.assertEqual(len(docs), 2)
        for doc in docs:
            resources = doc["spec"]["containers"][0]["resources"]
            self.assertEqual(resources["requests"], {"cpu": "4", "memory": "4Gi"})
            self.assertEqual(resources["limits"], {"cpu": "4", "memory": "4Gi"})

    def test_invalid_resources_fail_before_deploy(self) -> None:
        from src.benchmark.resources import PodResources

        config = replace(self.config(), resources=PodResources({"cpu": "16"}))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exceeds limit"):
                Iperf3Runner().render_manifest(config, Path(directory) / "m.yaml")

    def test_render_manifest_attaches_network(self) -> None:
        network = SecondaryNetwork(
            NetworkAttachment("secondary-a"), NetworkAttachment("secondary-b")
        )
        config = replace(self.config(), network=network)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "manifest.yaml"
            Iperf3Runner().render_manifest(config, destination)
            text = destination.read_text(encoding="utf-8")
        self.assertEqual(text.count("k8s.v1.cni.cncf.io/networks"), 2)
        self.assertIn("secondary-a", text)
        self.assertIn("secondary-b", text)


if __name__ == "__main__":
    unittest.main()
