from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

from src.backends.nixl.runner import (
    NixlRunner,
    _check_pipeline_depth,
    _env_assignments,
    _extra_flags,
    _minimum_total_buffer_size,
    _powers_of_two,
    _timed_bytes,
    _unknown_variables,
    _validate_args,
    _validate_env,
)
from src.benchmark.contracts import BackendResources, BenchmarkConfig
from src.benchmark.network import NetworkAttachment, SecondaryNetwork


class FakeProcess:
    def __init__(self, status: int | None = None) -> None:
        self.status = status

    def poll(self):
        return self.status

    def wait(self, timeout=None):
        self.status = 0
        return 0


class NixlRunnerTests(unittest.TestCase):
    def config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            backend="nixl", test_id="test-1", initiator_node="worker-a",
            target_node="worker-b", namespace="bench", image="example/nixl:latest",
            log_directory=Path("logs"), runs=3, parameters={
                "threads": 2, "iterations": 3, "warmup_iterations": 10,
                "block_size_start": 4 * 1024 * 1024,
                "block_size_max": 8 * 1024 * 1024,
                "batch_size_start": 1, "batch_size_max": 8,
            },
        )

    def test_validation_rejects_missing_or_invalid_warmup(self) -> None:
        runner = NixlRunner()
        with self.assertRaises(ValueError):
            runner.validate(replace(self.config(), parameters={}))
        with self.assertRaises(ValueError):
            runner.validate(replace(self.config(), parameters={"iterations": 3, "warmup_iterations": -1}))
        with self.assertRaisesRegex(ValueError, "iterations must be a positive integer"):
            runner.validate(replace(self.config(), parameters={"iterations": 0, "warmup_iterations": 1}))

    def test_buffer_calculation_and_execute(self) -> None:
        self.assertEqual(_minimum_total_buffer_size(3, 8, 1), 24)
        runner = NixlRunner(kubectl=Mock())
        def start_exec(namespace, pod, command, path):
            path.write_text(
                "Using address 10.0.0.1 port 12345\n" if pod == "server" else "ready\n",
                encoding="utf-8",
            )
            return FakeProcess()
        runner.kubectl.start_exec.side_effect = start_exec
        resources = BackendResources(names={
            "initiator_pod": "server", "target_pod": "client", "initiator_ip": "10.0.0.1",
        })
        with tempfile.TemporaryDirectory() as directory:
            output = runner.execute(self.config(), resources, Path(directory) / "raw" / "t1")
            self.assertTrue(output.is_dir())
            runner.cleanup(self.config(), BackendResources(manifest_path=Path("manifest.yaml")))
        runner.kubectl.delete_manifest.assert_called_once()

    def test_validation_rejects_invalid_run_timeout(self) -> None:
        runner = NixlRunner()
        config = self.config()
        for value in (0, -5, True, "10"):
            with self.assertRaises(ValueError):
                runner.validate(
                    replace(config, parameters={**config.parameters, "run_timeout_seconds": value})
                )

    def test_execute_times_out_and_terminates_processes(self) -> None:
        class HangingProcess(FakeProcess):
            terminated = False

            def wait(self, timeout=None):
                if timeout is not None and not self.terminated:
                    raise subprocess.TimeoutExpired("nixlbench", timeout)
                return super().wait(timeout)

            def send_signal(self, _signal):
                self.terminated = True

        processes = {}
        def start_exec(namespace, pod, command, path):
            path.write_text("Using address 10.0.0.1 port 12345\n", encoding="utf-8")
            processes[pod] = HangingProcess()
            return processes[pod]

        runner = NixlRunner(kubectl=Mock())
        runner.kubectl.start_exec.side_effect = start_exec
        resources = BackendResources(names={
            "initiator_pod": "server", "target_pod": "client", "initiator_ip": "10.0.0.1",
        })
        config = self.config()
        config = replace(config, parameters={**config.parameters, "run_timeout_seconds": 1})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                runner.execute(config, resources, Path(directory) / "raw" / "t1")
        self.assertTrue(all(process.terminated for process in processes.values()))

    def test_render_manifest_applies_image_pull_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "template.yaml"
            template.write_text(
                "__TEST_ID__ __CLIENT_NODE__ __SERVER_NODE__ __IMAGE__ "
                "policy=__IMAGE_PULL_POLICY__",
                encoding="utf-8",
            )
            destination = Path(directory) / "out.yaml"
            NixlRunner(manifest_template=template).render_manifest(
                replace(self.config(), image_pull_policy="Never"), destination
            )
            self.assertIn("policy=Never", destination.read_text(encoding="utf-8"))

    def test_gpu_false_removes_gpu_request_and_cuda_transport(self) -> None:
        runner = NixlRunner()
        with tempfile.TemporaryDirectory() as directory:
            gpu = Path(directory) / "gpu.yaml"
            cpu = Path(directory) / "cpu.yaml"
            runner.render_manifest(self.config(), gpu)
            runner.render_manifest(
                replace(self.config(), parameters={**self.config().parameters, "gpu": False}),
                cpu,
            )
            gpu_text = gpu.read_text(encoding="utf-8")
            cpu_text = cpu.read_text(encoding="utf-8")
        self.assertEqual(gpu_text.count("nvidia.com/gpu"), 4)
        self.assertIn("tcp,self,cuda_copy", gpu_text)
        self.assertNotIn("nvidia.com/gpu", cpu_text)
        self.assertNotIn("cuda_copy", cpu_text)
        self.assertEqual(cpu_text.count("value: tcp,self"), 2)
        self.assertIn("memory: 32Gi", cpu_text)  # other limits are kept
        # YAML 1.1 readers (kubectl) turn a bare n into false.
        self.assertIn('value: "n"', cpu_text)

    def test_validation_rejects_non_boolean_gpu(self) -> None:
        config = self.config()
        with self.assertRaises(ValueError):
            NixlRunner().validate(
                replace(config, parameters={**config.parameters, "gpu": "no"})
            )

    def network(self) -> SecondaryNetwork:
        return SecondaryNetwork(
            initiator=NetworkAttachment("cx7-a", "net1"),
            target=NetworkAttachment("cx7-b", "net2"),
        )

    def test_render_manifest_attaches_network_and_pins_ucx_devices(self) -> None:
        import yaml

        config = replace(self.config(), network=self.network())
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "manifest.yaml"
            NixlRunner().render_manifest(config, destination)
            docs = list(yaml.safe_load_all(destination.read_text(encoding="utf-8")))
        pods = {doc["metadata"]["name"]: doc for doc in docs}
        initiator = pods["nixlbench-server-test-1"]
        target = pods["nixlbench-client-test-1"]
        self.assertIn("cx7-a", initiator["metadata"]["annotations"]["k8s.v1.cni.cncf.io/networks"])
        self.assertIn("cx7-b", target["metadata"]["annotations"]["k8s.v1.cni.cncf.io/networks"])

        def env(pod):
            return {e["name"]: e["value"] for e in pod["spec"]["containers"][0]["env"]}

        self.assertEqual(env(initiator)["UCX_NET_DEVICES"], "net1")
        self.assertEqual(env(target)["UCX_NET_DEVICES"], "net2")
        self.assertEqual(env(initiator)["UCX_TLS"], "tcp,self,cuda_copy")  # untouched

    def test_no_network_leaves_ucx_devices_unset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "manifest.yaml"
            NixlRunner().render_manifest(self.config(), destination)
            text = destination.read_text(encoding="utf-8")
        self.assertNotIn("UCX_NET_DEVICES", text)
        self.assertNotIn("k8s.v1.cni.cncf.io", text)

    def test_deploy_uses_secondary_address_for_the_rendezvous(self) -> None:
        kubectl = Mock()
        kubectl.resource_exists.side_effect = lambda resource, namespace: (
            resource.startswith("network-attachment-definitions")
        )
        def output(args):
            if args[2] == "nixlbench-server-test-1":
                return '[{"interface": "net1", "ips": ["10.77.0.35"]}]'
            return '[{"interface": "net2", "ips": ["10.77.0.36"]}]'
        kubectl.output.side_effect = output
        resources = NixlRunner(kubectl=kubectl).deploy(
            replace(self.config(), network=self.network()), Path("manifest.yaml")
        )
        self.assertEqual(resources.names["initiator_ip"], "10.77.0.35")
        self.assertEqual(resources.names["initiator_address"], "10.77.0.35")
        self.assertEqual(resources.names["target_address"], "10.77.0.36")
        self.assertFalse(any(
            "podIP" in str(call) for call in kubectl.output.call_args_list
        ))

    def test_deploy_rejects_missing_attachment_before_creating_pods(self) -> None:
        kubectl = Mock()
        kubectl.resource_exists.return_value = False
        with self.assertRaisesRegex(ValueError, "not found"):
            NixlRunner(kubectl=kubectl).deploy(
                replace(self.config(), network=self.network()), Path("manifest.yaml")
            )
        kubectl.apply_manifest.assert_not_called()

    def test_resources_override_cpu_and_memory_but_keep_gpu(self) -> None:
        import yaml
        from src.benchmark.resources import PodResources

        config = replace(
            self.config(),
            resources=PodResources({"cpu": "4"}, {"cpu": "4", "memory": "16Gi"}),
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "manifest.yaml"
            NixlRunner().render_manifest(config, destination)
            docs = list(yaml.safe_load_all(destination.read_text(encoding="utf-8")))
        for doc in docs:
            resources = doc["spec"]["containers"][0]["resources"]
            self.assertEqual(resources["requests"]["cpu"], "4")
            self.assertEqual(resources["requests"]["memory"], "8Gi")  # template default kept
            self.assertEqual(resources["limits"], {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"})

    def execute_with(self, parameters, target_text="ready\n"):
        commands = {}
        def start_exec(namespace, pod, command, path):
            commands[pod] = list(command)
            path.write_text(
                "Using address 10.0.0.1 port 12345\n" if pod == "server" else target_text,
                encoding="utf-8",
            )
            return FakeProcess()
        runner = NixlRunner(kubectl=Mock())
        runner.kubectl.start_exec.side_effect = start_exec
        resources = BackendResources(names={
            "initiator_pod": "server", "target_pod": "client", "initiator_ip": "10.0.0.1",
        })
        config = replace(self.config(), parameters={**self.config().parameters, **parameters})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw" / "t1"
            try:
                runner.execute(config, resources, output)
            finally:
                import json
                metadata = json.loads(next(output.glob("*-command.json")).read_text(encoding="utf-8"))
        return commands, metadata

    def test_extra_args_and_env_reach_both_processes(self) -> None:
        commands, metadata = self.execute_with({
            "args": {"pipeline_depth": 16},
            "env": {"UCX_ZCOPY_THRESH": "1K"},
        })
        for pod in ("server", "client"):
            command = commands[pod]
            self.assertEqual(command[:3], ["env", "UCX_WARN_UNUSED_ENV_VARS=y", "UCX_ZCOPY_THRESH=1K"])
            self.assertEqual(command[3], "nixlbench")
            self.assertIn("--pipeline_depth=16", command)
            self.assertIn("--num_threads", command)  # managed flags are still passed
        self.assertEqual(commands["server"], commands["client"])
        self.assertIn("--pipeline_depth=16", metadata["nixl_args"])
        self.assertEqual(metadata["nixl_env"], {"UCX_ZCOPY_THRESH": "1K"})

    def test_without_args_and_env_the_command_is_unchanged(self) -> None:
        commands, metadata = self.execute_with({})
        self.assertEqual(commands["server"][0], "nixlbench")
        self.assertNotIn("env", commands["server"][:1])
        self.assertEqual(metadata["nixl_env"], {})
        self.assertFalse(any(a.startswith("--pipeline_depth") for a in commands["server"]))

    def test_unknown_ucx_variable_fails_the_run(self) -> None:
        warning = (
            "[1.0] [h:1:0] parser.c:2578 UCX  WARN  unused environment variable: "
            "UCX_ZCOPY_THRSH (maybe: UCX_ZCOPY_THRESH?)\n"
        )
        with self.assertRaisesRegex(RuntimeError, "UCX_ZCOPY_THRSH"):
            self.execute_with({"env": {"UCX_ZCOPY_THRSH": "1K"}}, target_text=warning)

    def test_validation_rejects_unknown_parallelism(self) -> None:
        config = self.config()
        for value in ("instances", "", True, 2):
            with self.assertRaisesRegex(ValueError, "parallelism"):
                NixlRunner().validate(
                    replace(config, parameters={**config.parameters, "parallelism": value})
                )
        for value in ("threads", "processes"):
            NixlRunner().validate(
                replace(config, parameters={**config.parameters, "parallelism": value})
            )

    def execute_processes(self, threads: int, fail_on_call: int | None = None, with_rows: bool = False):
        import src.backends.nixl.runner as runner_module

        commands = []
        class Failing(FakeProcess):
            def wait(self, timeout=None):
                return 1

        def start_exec(namespace, pod, command, path):
            text = "Using address 10.0.0.1 port 12345\n"
            if with_rows and pod == "server":
                block = int(command[command.index("--max_block_size") + 1])
                batch = int(command[command.index("--max_batch_size") + 1])
                text += f"{block:<20}{batch:<15}2.448327       428.3\n"
            path.write_text(text, encoding="utf-8")
            commands.append((pod, command, path.name))
            return Failing() if fail_on_call == len(commands) else FakeProcess()

        runner = NixlRunner(kubectl=Mock())
        runner.kubectl.start_exec.side_effect = start_exec
        resources = BackendResources(names={
            "initiator_pod": "server", "target_pod": "client", "initiator_ip": "10.0.0.1",
        })
        config = self.config()
        config = replace(config, parameters={
            **config.parameters, "threads": threads, "parallelism": "processes",
        })
        original = runner_module.PEER_GRACE_SECONDS
        runner_module.PEER_GRACE_SECONDS = 0.0
        error = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "raw" / f"t{threads}"
                try:
                    runner.execute(config, resources, output)
                except RuntimeError as caught:
                    error = caught
                metadata = json.loads(next(output.glob("*-command.json")).read_text(encoding="utf-8"))
        finally:
            runner_module.PEER_GRACE_SECONDS = original
        return commands, metadata, error

    def flag(self, command, name):
        return command[command.index(name) + 1]

    def test_processes_run_one_group_of_single_thread_pairs_per_point(self) -> None:
        commands, metadata, error = self.execute_processes(3)
        self.assertIsNone(error)
        # 2 block sizes x 4 batch sizes, 3 pairs each: initiators first, then targets.
        self.assertEqual(len(commands), 8 * 6)
        for start in range(0, len(commands), 6):
            group = commands[start:start + 6]
            self.assertEqual([pod for pod, _, _ in group], ["server"] * 3 + ["client"] * 3)
            blocks = {self.flag(c, "--max_block_size") for _, c, _ in group}
            batches = {self.flag(c, "--max_batch_size") for _, c, _ in group}
            self.assertEqual(len(blocks), 1)
            self.assertEqual(len(batches), 1)
            for _, command, _ in group:
                self.assertEqual(self.flag(command, "--num_threads"), "1")
                self.assertEqual(self.flag(command, "--start_block_size"), self.flag(command, "--max_block_size"))
                self.assertEqual(self.flag(command, "--start_batch_size"), self.flag(command, "--max_batch_size"))
                self.assertEqual(
                    int(self.flag(command, "--total_buffer_size")),
                    int(self.flag(command, "--max_block_size")) * int(self.flag(command, "--max_batch_size")),
                )
            ports = [self.flag(c, "--asio_port") for p, c, _ in group if p == "server"]
            self.assertEqual(ports, [self.flag(c, "--asio_port") for p, c, _ in group if p == "client"])
        all_ports = [self.flag(c, "--asio_port") for p, c, _ in commands if p == "server"]
        self.assertEqual(len(set(all_ports)), len(all_ports))
        self.assertEqual(metadata["parallelism"], "processes")
        self.assertEqual(metadata["processes"], 3)
        self.assertEqual(len(metadata["points"]), 8)
        self.assertEqual(
            [(p["block_size"], p["batch_size"]) for p in metadata["points"][:5]],
            [(4 << 20, 1), (4 << 20, 2), (4 << 20, 4), (4 << 20, 8), (8 << 20, 1)],
        )
        first = metadata["points"][0]["files"]
        self.assertEqual(
            [f["initiator_file"] for f in first],
            [f"test-1-t3-run1-b{4 << 20}-n1-p{i}-initiator.log" for i in (1, 2, 3)],
        )
        self.assertEqual(metadata["initiator_status"], 0)
        self.assertEqual(metadata["parameters"]["num_threads"], 1)

    def test_a_failing_point_stops_the_sweep_and_names_the_point(self) -> None:
        # Call 3 is the second point's first initiator (2 pairs per point).
        commands, metadata, error = self.execute_processes(2, fail_on_call=5)
        self.assertIsNotNone(error)
        self.assertRegex(str(error), r"initiator 1=1.*block_size=4M batch_size=2")
        self.assertEqual(len(commands), 8)  # two points, four processes each
        self.assertEqual(metadata["initiator_status"], 1)

    def test_threads_mode_is_one_pair_with_the_whole_sweep(self) -> None:
        commands, metadata = self.execute_with({})
        self.assertEqual(len(commands), 2)
        self.assertEqual(self.flag(commands["server"], "--start_block_size"), str(4 << 20))
        self.assertEqual(self.flag(commands["server"], "--max_block_size"), str(8 << 20))
        self.assertEqual(metadata["parallelism"], "threads")
        self.assertNotIn("points", metadata)

    def test_pipeline_depth_check_uses_one_thread_per_process(self) -> None:
        config = self.pipeline_config(1280)
        with self.assertRaisesRegex(ValueError, "8 thread"):
            NixlRunner().validate(config)
        NixlRunner().validate(
            replace(config, parameters={**config.parameters, "parallelism": "processes"})
        )

    def test_timed_bytes_follow_nixlbench_iteration_rules(self) -> None:
        mib = 1 << 20
        # Blocks up to 1 MiB run all (rounded up) iterations, larger ones a sixteenth.
        self.assertEqual(_timed_bytes(12800, 16, 1, mib, 2), 12800 * mib * 2)
        self.assertEqual(_timed_bytes(12800, 16, 1, 2 * mib, 1), 800 * 2 * mib)
        self.assertEqual(_timed_bytes(100, 16, 1, mib, 1), 112 * mib)  # rounded up to 16
        self.assertEqual(_timed_bytes(100, 16, 2, mib, 1), 128 * mib)  # to 2 x 16
        self.assertEqual(_timed_bytes(12800, 8, 1, 4 * mib, 1), 1600 * 4 * mib)

    def test_points_record_timed_bytes_and_when_each_result_appeared(self) -> None:
        commands, metadata, error = self.execute_processes(2, with_rows=True)
        self.assertIsNone(error)
        mib = 1 << 20
        first = metadata["points"][0]
        self.assertEqual(first["timed_bytes"], _timed_bytes(3, 16, 1, 4 * mib, 1))
        seen = [entry["row_seen_s"] for entry in first["files"]]
        self.assertTrue(all(isinstance(value, float) and value >= 0 for value in seen))
        # Threads mode needs none of this.
        _, threads_metadata = self.execute_with({})
        self.assertNotIn("points", threads_metadata)

    def test_sweep_points_double_like_nixlbench(self) -> None:
        self.assertEqual(_powers_of_two(1, 8), [1, 2, 4, 8])
        self.assertEqual(_powers_of_two(256 << 10, 1 << 20), [256 << 10, 512 << 10, 1 << 20])
        self.assertEqual(_powers_of_two(4, 4), [4])

    def test_validation_checks_iterations(self) -> None:
        config = self.config()
        for value in (0, -1, True, "2", 1.5):
            with self.assertRaisesRegex(ValueError, "iterations must be a positive integer"):
                NixlRunner().validate(
                    replace(config, parameters={**config.parameters, "iterations": value})
                )
        NixlRunner().validate(replace(config, parameters={**config.parameters, "iterations": 3}))

    def test_validation_checks_the_backend(self) -> None:
        config = self.config()
        for value in ("ucx", "GDS", "", True):
            with self.assertRaisesRegex(ValueError, "backend must be one of"):
                NixlRunner().validate(
                    replace(config, parameters={**config.parameters, "backend": value})
                )
        for value in ("UCX", "LIBFABRIC", "Mooncake"):
            NixlRunner().validate(
                replace(config, parameters={**config.parameters, "backend": value})
            )

    def test_mooncake_rejects_progress_thread_args(self) -> None:
        for name, value in (("enable_pt", True), ("progress_threads", 2)):
            with self.assertRaisesRegex(ValueError, "not supported by the Mooncake"):
                _validate_args({name: value}, backend="Mooncake")
        _validate_args({"enable_pt": True, "progress_threads": 2}, backend="LIBFABRIC")

    def test_env_names_follow_the_backend(self) -> None:
        _validate_env({"FI_TCP_IFACE": "x", "NIXL_FOO": 1}, network_set=False, backend="LIBFABRIC")
        _validate_env({"MC_FORCE_TCP": 1}, network_set=False, backend="Mooncake")
        with self.assertRaisesRegex(ValueError, "FI_, NIXL_ or OMP_"):
            _validate_env({"UCX_TLS": "tcp"}, network_set=False, backend="LIBFABRIC")
        with self.assertRaisesRegex(ValueError, "MC_, NIXL_ or OMP_"):
            _validate_env({"FI_PROVIDER": "tcp"}, network_set=False, backend="Mooncake")

    def test_network_variables_are_managed_per_backend(self) -> None:
        for backend, name in (
            ("UCX", "UCX_NET_DEVICES"),
            ("LIBFABRIC", "FI_TCP_IFACE"),
            ("Mooncake", "NIXL_MOONCAKE_IP_ADDR"),
        ):
            _validate_env({name: "x"}, network_set=False, backend=backend)
            with self.assertRaisesRegex(ValueError, "managed by the network section"):
                _validate_env({name: "x"}, network_set=True, backend=backend)
        # Another backend's pin is not managed (and not even a valid name).
        _validate_env({"NIXL_MOONCAKE_IP_ADDR": "x"}, network_set=True, backend="LIBFABRIC")

    def execute_backend(self, backend, *, network=False, env=None, args=None):
        commands = {}
        def start_exec(namespace, pod, command, path):
            commands[pod] = list(command)
            path.write_text("Using address 10.0.0.1 port 12345\n", encoding="utf-8")
            return FakeProcess()
        runner = NixlRunner(kubectl=Mock())
        runner.kubectl.start_exec.side_effect = start_exec
        names = {"initiator_pod": "server", "target_pod": "client", "initiator_ip": "10.0.0.1"}
        if network:
            names.update(initiator_address="10.77.0.35", target_address="10.77.0.36")
        parameters = {**self.config().parameters, "backend": backend}
        if env is not None:
            parameters["env"] = env
        if args is not None:
            parameters["args"] = args
        config = replace(self.config(), parameters=parameters, network=self.network() if network else None)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw" / "t1"
            try:
                runner.execute(config, BackendResources(names=names), output)
            finally:
                metadata = json.loads(next(output.glob("*-command.json")).read_text(encoding="utf-8"))
        return commands, metadata

    def test_default_backend_is_ucx_and_unchanged(self) -> None:
        commands, metadata = self.execute_with({})
        self.assertEqual(self.flag(commands["server"], "--backend"), "UCX")
        self.assertEqual(commands["server"][0], "nixlbench")
        self.assertEqual(metadata["nixl_backend"], "UCX")

    def test_libfabric_runs_on_tcp_and_is_pinned_to_the_interface(self) -> None:
        commands, metadata = self.execute_backend("LIBFABRIC", network=True)
        for pod, interface in (("server", "net1"), ("client", "net2")):
            command = commands[pod]
            self.assertEqual(command[:3], ["env", "FI_PROVIDER=tcp", f"FI_TCP_IFACE={interface}"])
            self.assertEqual(command[3], "nixlbench")
            self.assertEqual(self.flag(command, "--backend"), "LIBFABRIC")
            self.assertFalse(any(part.startswith("UCX_") for part in command))
        self.assertEqual(metadata["nixl_backend"], "LIBFABRIC")
        self.assertEqual(metadata["parameters"]["backend"], "LIBFABRIC")

    def test_libfabric_always_runs_with_a_progress_thread(self) -> None:
        commands, metadata = self.execute_backend("LIBFABRIC", args={"pipeline_depth": 2})
        self.assertIn("--enable_pt=true", commands["server"])
        self.assertIn("--pipeline_depth=2", commands["server"])
        self.assertEqual(commands["server"], commands["client"])
        self.assertIn("--enable_pt=true", metadata["nixl_args"])
        _validate_args({"enable_pt": True}, backend="LIBFABRIC")
        with self.assertRaisesRegex(ValueError, "enable_pt must be true for the LIBFABRIC"):
            _validate_args({"enable_pt": False}, backend="LIBFABRIC")
        commands, _ = self.execute_backend("UCX")
        self.assertNotIn("--enable_pt=true", commands["server"])

    def test_libfabric_without_network_only_forces_tcp(self) -> None:
        commands, _ = self.execute_backend("LIBFABRIC")
        self.assertEqual(commands["server"][:3], ["env", "FI_PROVIDER=tcp", "nixlbench"])

    def test_mooncake_forces_tcp_and_advertises_each_sides_address(self) -> None:
        commands, _ = self.execute_backend("Mooncake", network=True)
        self.assertEqual(
            commands["server"][:5],
            ["env", "MC_FORCE_TCP=1", "MC_NUM_TCP_IO_THREADS=2", "NIXL_MOONCAKE_IP_ADDR=10.77.0.35", "nixlbench"],
        )
        self.assertEqual(
            commands["client"][:5],
            ["env", "MC_FORCE_TCP=1", "MC_NUM_TCP_IO_THREADS=2", "NIXL_MOONCAKE_IP_ADDR=10.77.0.36", "nixlbench"],
        )
        self.assertEqual(self.flag(commands["server"], "--backend"), "Mooncake")

    def test_mooncake_io_threads_follow_the_thread_count(self) -> None:
        commands, metadata = self.execute_backend("Mooncake")  # config() runs 2 threads
        self.assertIn("MC_NUM_TCP_IO_THREADS=2", commands["server"])
        self.assertEqual(metadata["nixl_env_applied"]["target"]["MC_NUM_TCP_IO_THREADS"], "2")
        commands, _ = self.execute_backend("Mooncake", env={"MC_NUM_TCP_IO_THREADS": 6})
        self.assertIn("MC_NUM_TCP_IO_THREADS=6", commands["server"])
        self.assertNotIn("MC_NUM_TCP_IO_THREADS=2", commands["server"])
        # Other backends do not get it, and processes mode runs one thread per process.
        commands, _ = self.execute_backend("LIBFABRIC")
        self.assertFalse(any("MC_NUM" in part for part in commands["server"]))
        from src.backends.nixl.runner import _environment

        config = self.config()
        config = replace(config, parameters={
            **config.parameters, "backend": "Mooncake", "threads": 5, "parallelism": "processes",
        })
        resources = BackendResources(names={})
        self.assertEqual(_environment(config, resources, "target")["MC_NUM_TCP_IO_THREADS"], "1")
        config = replace(config, parameters={**config.parameters, "parallelism": "threads"})
        self.assertEqual(_environment(config, resources, "target")["MC_NUM_TCP_IO_THREADS"], "5")

    def test_env_placeholders_follow_the_run_and_the_side(self) -> None:
        env = {"MC_TENT_CONF": '{"w": {threads}, "a": "{address}"}'}
        commands, metadata = self.execute_backend("Mooncake", network=True, env=env)
        self.assertIn('MC_TENT_CONF={"w": 2, "a": "10.77.0.35"}', commands["server"])
        self.assertIn('MC_TENT_CONF={"w": 2, "a": "10.77.0.36"}', commands["client"])
        self.assertEqual(
            metadata["nixl_env_applied"]["target"]["MC_TENT_CONF"], '{"w": 2, "a": "10.77.0.36"}'
        )

    def test_threads_placeholder_takes_a_multiplier(self) -> None:
        env = {"MC_TENT_CONF": '{"w": {threads}, "c": {threads*3}}'}
        commands, metadata = self.execute_backend("Mooncake", network=True, env=env)
        self.assertIn('MC_TENT_CONF={"w": 2, "c": 6}', commands["server"])
        self.assertEqual(
            metadata["nixl_env_applied"]["target"]["MC_TENT_CONF"], '{"w": 2, "c": 6}'
        )

    def test_address_placeholder_needs_a_network(self) -> None:
        config = self.config()
        with self.assertRaisesRegex(ValueError, r"uses \{address\}, which needs a network"):
            NixlRunner().validate(replace(config, parameters={
                **config.parameters, "backend": "Mooncake", "env": {"MC_TENT_CONF": "{address}"},
            }))

    def test_tent_replaces_the_forced_tcp_default(self) -> None:
        commands, _ = self.execute_backend("Mooncake", env={"MC_USE_TENT": 1})
        self.assertIn("MC_USE_TENT=1", commands["server"])
        self.assertFalse(any(part.startswith("MC_FORCE_TCP") for part in commands["server"]))
        commands, _ = self.execute_backend("Mooncake", env={"MC_USE_TENT": 0})
        self.assertIn("MC_FORCE_TCP=1", commands["server"])

    def test_profile_env_overrides_the_backend_defaults(self) -> None:
        commands, _ = self.execute_backend("LIBFABRIC", env={"FI_PROVIDER": "sockets", "FI_LOG_LEVEL": "warn"})
        self.assertIn("FI_PROVIDER=sockets", commands["server"])
        self.assertNotIn("FI_PROVIDER=tcp", commands["server"])
        self.assertIn("FI_LOG_LEVEL=warn", commands["server"])
        self.assertFalse(any("UCX_WARN_UNUSED_ENV_VARS" in part for part in commands["server"]))

    def test_other_backends_do_not_get_the_ucx_device_pin(self) -> None:
        import yaml

        config = replace(
            self.config(), network=self.network(),
            parameters={**self.config().parameters, "backend": "LIBFABRIC"},
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "manifest.yaml"
            NixlRunner().render_manifest(config, destination)
            self.assertNotIn("UCX_NET_DEVICES", destination.read_text(encoding="utf-8"))

    def execute_teardown(self, *, rows: bool = True, error: str = "ASIO Read head failed: End of file",
                         initiator_status: int = 0, threads: int = 3, failing: int = 2):
        """A processes run where target `failing` (1-based) of the first point exits 134."""

        import src.backends.nixl.runner as runner_module

        class Exit(FakeProcess):
            def __init__(self, status):
                super().__init__()
                self.code = status

            def wait(self, timeout=None):
                return self.code

        calls = []
        def start_exec(namespace, pod, command, path):
            calls.append(pod)
            block = int(command[command.index("--max_block_size") + 1])
            batch = int(command[command.index("--max_batch_size") + 1])
            if pod == "server":
                text = "Using address 10.0.0.1 port 12345\n"
                if rows:
                    text += f"{block:<20}{batch:<15}2.448327       428.3\n"
                path.write_text(text, encoding="utf-8")
                return Exit(initiator_status if len(calls) == 1 else 0)
            path.write_text("ready\nterminate called\n  what():  " + error + "\n", encoding="utf-8")
            first_targets = [i for i, pod_ in enumerate(calls) if pod_ == "client"]
            is_failing = len(first_targets) == failing and len(calls) <= 2 * threads
            return Exit(134 if is_failing else 0)

        runner = NixlRunner(kubectl=Mock())
        runner.kubectl.start_exec.side_effect = start_exec
        resources = BackendResources(names={
            "initiator_pod": "server", "target_pod": "client", "initiator_ip": "10.0.0.1",
        })
        config = self.config()
        config = replace(config, parameters={
            **config.parameters, "threads": threads, "parallelism": "processes",
            "block_size_start": 4 << 20, "block_size_max": 4 << 20,
            "batch_size_start": 1, "batch_size_max": 1,
        })
        original = runner_module.PEER_GRACE_SECONDS
        runner_module.PEER_GRACE_SECONDS = 0.0
        error_raised = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "raw" / f"t{threads}"
                try:
                    runner.execute(config, resources, output)
                except RuntimeError as caught:
                    error_raised = caught
                metadata = json.loads(next(output.glob("*-command.json")).read_text(encoding="utf-8"))
        finally:
            runner_module.PEER_GRACE_SECONDS = original
        return metadata, error_raised

    def test_startup_failure_reports_the_reason_from_the_log(self) -> None:
        class Dead(FakeProcess):
            def poll(self):
                return 1

        def start_exec(namespace, pod, command, path):
            path.write_text(
                "Using address 10.0.0.1 port 12345\n"
                "E1016 15:36:22.171194       7 nixl_agent.cpp:318] getPluginParams: backend "
                "'Mooncake' not found\nNIXL: createBackend failed! (NIXL_ERR_NOT_FOUND)\n",
                encoding="utf-8",
            )
            return Dead()

        runner = NixlRunner(kubectl=Mock())
        runner.kubectl.start_exec.side_effect = start_exec
        resources = BackendResources(names={
            "initiator_pod": "server", "target_pod": "client", "initiator_ip": "10.0.0.1",
        })
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeError, r"exited before becoming ready: E1016 .*backend 'Mooncake' not found"
            ):
                runner.execute(self.config(), resources, Path(directory) / "raw" / "t1")

    def test_target_teardown_race_after_a_complete_initiator_is_tolerated(self) -> None:
        metadata, error = self.execute_teardown()
        self.assertIsNone(error)
        self.assertEqual(metadata["target_status"], 0)
        (entry,) = metadata["tolerated_target_exits"]
        self.assertEqual(entry["process"], "target 2")
        self.assertEqual(entry["status"], 134)
        self.assertIn("-p2-target.log", entry["log"])

    def test_teardown_error_without_the_initiator_rows_is_a_failure(self) -> None:
        _, error = self.execute_teardown(rows=False)
        self.assertRegex(str(error), r"target 2=134")

    def test_other_target_errors_are_failures(self) -> None:
        _, error = self.execute_teardown(error="ASIO Receive timeout")
        self.assertRegex(str(error), r"target 2=134")

    def test_a_failing_initiator_is_never_tolerated(self) -> None:
        _, error = self.execute_teardown(initiator_status=1)
        self.assertRegex(str(error), r"initiator 1=1")

    def test_tolerance_helper_needs_every_expected_row(self) -> None:
        from src.backends.nixl.runner import _expected_rows, _tolerated_target_exits

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initiator = root / "i.log"
            target = root / "t.log"
            target.write_text("what():  ASIO Read head failed: End of file\n", encoding="utf-8")
            expected = _expected_rows((1 << 20, 2 << 20, 1, 2))
            self.assertEqual(len(expected), 4)
            initiator.write_text(
                "".join(f"{b:<20}{n:<15}1.5      10.0\n" for b, n in expected[:3]), encoding="utf-8"
            )
            statuses = {"initiator": 0, "target": 134}
            self.assertEqual(_tolerated_target_exits(statuses, [initiator], [target], expected), [])
            initiator.write_text(
                "".join(f"{b:<20}{n:<15}1.5      10.0\n" for b, n in expected), encoding="utf-8"
            )
            tolerated = _tolerated_target_exits(statuses, [initiator], [target], expected)
            self.assertEqual([t["process"] for t in tolerated], ["target"])

    def pipeline_config(self, warmup: int):
        config = self.config()
        return replace(
            config,
            parameters={
                **config.parameters,
                "iterations": 12800,
                "threads_spec": "1:8:7",
                "block_size_max": 8 * 1024 * 1024,
                "warmup_iterations": warmup,
                "args": {"pipeline_depth": 16},
            },
        )

    def test_pipeline_depth_is_validated_against_the_whole_sweep(self) -> None:
        with self.assertRaisesRegex(ValueError, "8 thread"):
            NixlRunner().validate(self.pipeline_config(1280))
        NixlRunner().validate(self.pipeline_config(2048))

    def test_failure_does_not_wait_for_the_other_side(self) -> None:
        import src.backends.nixl.runner as runner_module

        class Failing(FakeProcess):
            def wait(self, timeout=None):
                return 1

        class Stuck(FakeProcess):
            terminated = False

            def wait(self, timeout=None):
                if self.terminated:
                    return -15
                raise subprocess.TimeoutExpired("nixlbench", timeout)

            def send_signal(self, _signal):
                self.terminated = True

        stuck = Stuck()
        started = {}
        def start_exec(namespace, pod, command, path):
            path.write_text(
                "Using address 10.0.0.1 port 12345\n" if pod == "server" else "waiting\n",
                encoding="utf-8",
            )
            started[pod] = Failing() if pod == "server" else stuck
            if pod == "server":
                path.write_text(
                    "Using address 10.0.0.1 port 12345\nError: descriptor count (16) is not evenly "
                    "divisible by pipeline depth (10)\n",
                    encoding="utf-8",
                )
            return started[pod]

        runner = NixlRunner(kubectl=Mock())
        runner.kubectl.start_exec.side_effect = start_exec
        resources = BackendResources(names={
            "initiator_pod": "server", "target_pod": "client", "initiator_ip": "10.0.0.1",
        })
        original = runner_module.PEER_GRACE_SECONDS
        runner_module.PEER_GRACE_SECONDS = 0.0
        try:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(
                    RuntimeError, r"initiator=1, target=-15\): Error: descriptor count"
                ):
                    runner.execute(self.config(), resources, Path(directory) / "raw" / "t1")
        finally:
            runner_module.PEER_GRACE_SECONDS = original
        self.assertTrue(stuck.terminated)

    def test_validation_covers_args_and_env(self) -> None:
        config = self.config()
        with self.assertRaisesRegex(ValueError, "pipeline_depth"):
            NixlRunner().validate(replace(config, parameters={**config.parameters, "args": {"pipeline_depth": 0}}))
        with self.assertRaisesRegex(ValueError, "managed by the network section"):
            NixlRunner().validate(replace(
                config, network=self.network(),
                parameters={**config.parameters, "env": {"UCX_NET_DEVICES": "net1"}},
            ))

    def test_render_manifest_rejects_leftover_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "template.yaml"
            template.write_text(
                "__TEST_ID__ __CLIENT_NODE__ __SERVER_NODE__ __IMAGE__ __IMAGE_PULL_POLICY__", encoding="utf-8"
            )
            runner = NixlRunner(manifest_template=template)
            config = replace(self.config(), image="registry/__TEST_ID__:tag")
            with self.assertRaisesRegex(ValueError, "still contains"):
                runner.render_manifest(config, Path(directory) / "out.yaml")


class NixlArgsTests(unittest.TestCase):
    def test_valid_args_become_gflags_style_flags(self) -> None:
        args = {"pipeline_depth": 16, "enable_pt": True, "op_type": "READ", "use_hugepages": False}
        _validate_args(args)
        self.assertEqual(
            _extra_flags(args),
            ["--pipeline_depth=16", "--enable_pt=true", "--op_type=READ", "--use_hugepages=false"],
        )
        _validate_args(None)
        self.assertEqual(_extra_flags(None), [])

    def test_invalid_args_are_rejected(self) -> None:
        for args, message in (
            ({"pipeline_depth": 0}, "integer >= 1"),
            ({"pipeline_depth": True}, "integer >= 1"),
            ({"pipeline_depth": "8"}, "integer >= 1"),
            ({"progress_threads": -1}, "integer >= 0"),
            ({"enable_pt": 1}, "true or false"),
            ({"op_type": "COPY"}, "one of: READ, WRITE"),
            ({"num_threads": 4}, "managed by the profile"),
            ({"total_buffer_size": 1}, "managed by the profile"),
            ({"bogus_flag": 1}, "not supported"),
            ([1, 2], "must be a mapping"),
        ):
            with self.assertRaisesRegex(ValueError, message, msg=args):
                _validate_args(args)


class NixlEnvTests(unittest.TestCase):
    def test_valid_env(self) -> None:
        _validate_env({"UCX_ZCOPY_THRESH": "1K", "UCX_TCP_NODELAY": False, "OMP_PROC_BIND": "close", "UCX_MAX_RNDV_RAILS": 1}, network_set=False)
        _validate_env(None, network_set=True)

    def test_invalid_env(self) -> None:
        for env, network_set, message in (
            ({"ucx_tcp_nodelay": "n"}, False, "variable name in capitals"),
            ({"PATH": "/x"}, False, "variable name in capitals"),
            ({"UCX_NET_DEVICES": "net1"}, True, "managed by the network section"),
            ({"UCX_TLS": ["tcp"]}, False, "string, number or boolean"),
            ({"UCX_TLS": "a\nb"}, False, "string, number or boolean"),
        ):
            with self.assertRaisesRegex(ValueError, message, msg=env):
                _validate_env(env, network_set=network_set)
        _validate_env({"UCX_NET_DEVICES": "mlx5_0:1"}, network_set=False)

    def test_assignments_always_ask_ucx_to_report_unknown_variables(self) -> None:
        self.assertEqual(_env_assignments(None), [])
        self.assertEqual(_env_assignments({}), [])
        self.assertEqual(
            _env_assignments({"UCX_ZCOPY_THRESH": "1K", "UCX_TCP_NODELAY": False}),
            ["UCX_WARN_UNUSED_ENV_VARS=y", "UCX_ZCOPY_THRESH=1K", "UCX_TCP_NODELAY=n"],
        )
        # An explicit setting wins over the default.
        self.assertEqual(
            _env_assignments({"UCX_WARN_UNUSED_ENV_VARS": False}),
            ["UCX_WARN_UNUSED_ENV_VARS=n"],
        )

    def test_unknown_variables_are_read_from_ucx_warnings(self) -> None:
        log = (
            "[1.0] [host:1:0] parser.c:2578 UCX  WARN  unused environment variable: "
            "UCX_TCP_TX_SEG_SZ (maybe: UCX_TCP_TX_SEG_SIZE, UCX_TCP_RX_SEG_SIZE?)\n"
            "[1.0] [host:1:0] parser.c:2578 UCX  WARN  (set UCX_WARN_UNUSED_ENV_VARS=n to suppress this warning)\n"
        )
        self.assertEqual(_unknown_variables(log, log, "clean log"), ["UCX_TCP_TX_SEG_SZ"])
        self.assertEqual(_unknown_variables("nothing"), [])


class NixlPipelineDepthTests(unittest.TestCase):
    def check(self, **overrides) -> None:
        values = dict(
            depth=16, iterations=12800, warmup_iterations=2048, thread_values=(1, 8),
            max_block_size=8 * 1024 * 1024,
        )
        _check_pipeline_depth(**{**values, **overrides})

    def test_reported_case_is_rejected_with_the_fix(self) -> None:
        # 1280 warmup / 8 threads / 16 = 10 per thread for blocks above 1 MiB.
        with self.assertRaises(ValueError) as caught:
            self.check(warmup_iterations=1280)
        message = str(caught.exception)
        self.assertIn("pipeline_depth 16 exceeds the 10 warmup iterations for blocks above 1 MiB", message)
        self.assertIn("8 thread(s)", message)
        self.assertIn("at most 10", message)
        self.assertIn("at least 2048", message)

    def test_depth_within_the_per_thread_counts_is_accepted(self) -> None:
        self.check()  # 2048 / 8 / 16 = 16
        self.check(warmup_iterations=1280, depth=10)
        self.check(depth=1, warmup_iterations=1)
        self.check(depth=0)

    def test_timed_iterations_are_checked_too(self) -> None:
        with self.assertRaisesRegex(ValueError, "8 timed iterations for blocks above 1 MiB"):
            self.check(iterations=1024, depth=16, warmup_iterations=2048, thread_values=(1, 8))

    def test_without_large_blocks_the_division_by_the_factor_does_not_apply(self) -> None:
        # 128 iterations / 8 threads = 16 per thread, enough for depth 16.
        self.check(iterations=128, warmup_iterations=128, thread_values=(8,), max_block_size=1024 * 1024)
        with self.assertRaisesRegex(ValueError, "exceeds the 16 timed iterations"):
            self.check(iterations=128, warmup_iterations=128, depth=17, thread_values=(8,), max_block_size=1024 * 1024)

    def test_zero_warmup_is_not_checked_and_counts_are_rounded_up(self) -> None:
        self.check(warmup_iterations=0)
        self.check(iterations=1, warmup_iterations=1, thread_values=(1,), max_block_size=1024, depth=16)
        # 100 rounds up to 112 for one thread; 112 / 16 = 7 for large blocks.
        with self.assertRaisesRegex(ValueError, "exceeds the 7 "):
            self.check(iterations=100, warmup_iterations=0, depth=8, thread_values=(1,))

    def test_custom_large_block_factor_is_used(self) -> None:
        self.check(large_blk_iter_ftr=1, warmup_iterations=128, iterations=128, thread_values=(8,), depth=16)


if __name__ == "__main__":
    unittest.main()
