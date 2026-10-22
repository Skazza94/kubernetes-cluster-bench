from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock

from src.backends.iperf3.parser import Iperf3Parser
from src.backends.nixl.parser import NixlParser
from src.backends.registry import default_registry
from src.benchmark.cpu import (
    SAMPLER_SCRIPT,
    CpuSampler,
    cleanup_problems,
    cpu_metadata,
    cpu_usage,
    read_samples,
    read_thread_sweeps,
    stop_command,
    thread_usage,
)
from src.graphing.cli import main as graph_main
from src.graphing.contracts import Observation
from src.graphing.cpu import cpu_observations


def write_samples(path: Path, cores: float, throttled_share: float = 0.0, seconds: int = 20) -> None:
    """One sample per second of a pod busy on ``cores`` cores."""

    lines = []
    for second in range(seconds + 1):
        periods = second * 10
        lines.append(
            f"{second:.4f} {int(second * cores * 1e6)} 0 {periods} {int(periods * throttled_share)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class LineKubectl:
    """Streams fixed sampler lines, as the in-pod sampler would."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def start_stream(self, namespace, pod, command):
        script = "; ".join(f"echo '{line}'; sleep 0.02" for line in self.lines)
        return subprocess.Popen(["sh", "-c", script], stdout=subprocess.PIPE)


class CpuSamplesTests(unittest.TestCase):
    def test_usage_is_interpolated_over_the_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpu.log"
            write_samples(path, cores=2.5, throttled_share=0.2)
            path.write_text(path.read_text() + "garbage line\n", encoding="utf-8")
            samples = read_samples(path)
        self.assertEqual(len(samples), 21)
        usage = cpu_usage(samples, 2.5, 7.25)
        self.assertAlmostEqual(usage["cores"], 2.5)
        self.assertAlmostEqual(usage["throttled"], 0.2, places=1)

    def test_a_window_outside_the_samples_has_no_usage(self) -> None:
        samples = [(1.0, 0.0, 0.0, 0.0, 0.0), (2.0, 1.0, 0.0, 0.0, 0.0)]
        self.assertIsNone(cpu_usage(samples, 0.5, 1.5))
        self.assertIsNone(cpu_usage(samples, 1.5, 2.5))
        self.assertIsNone(cpu_usage(samples, 1.5, 1.5))
        self.assertIsNone(cpu_usage(samples[:1], 1.0, 1.0))

    def test_metadata_has_both_sides_and_the_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_samples(run / "i.log", cores=1.5)
            write_samples(run / "t.log", cores=3.0)
            details = cpu_metadata(run, {"cpu_files": {"initiator": "i.log", "target": "t.log"}}, 4.0, 9.0)
            self.assertEqual(cpu_metadata(run, None, 4.0, 9.0), {})
            self.assertEqual(cpu_metadata(run, {"cpu_files": {"initiator": "missing.log"}}, 4.0, 9.0), {})
        self.assertAlmostEqual(details["cpu_cores_initiator"], 1.5)
        self.assertAlmostEqual(details["cpu_cores_target"], 3.0)
        self.assertAlmostEqual(details["cpu_window_s"], 5.0)


class ThreadUsageTests(unittest.TestCase):
    def test_threads_are_measured_between_the_sweeps_in_the_window(self) -> None:
        sweeps = [
            (0.0, {1: (1, 0, "main"), 2: (1, 0, "worker")}),
            (2.0, {1: (1, 100, "main"), 2: (1, 200, "worker")}),
            (4.0, {1: (1, 300, "main"), 2: (1, 400, "worker"), 3: (1, 100, "late")}),
            (6.0, {1: (1, 500, "main"), 2: (1, 401, "worker"), 3: (1, 300, "late")}),
        ]
        threads = thread_usage(100.0, sweeps, 1.9, 6.1)
        # 2 s to 6 s: main used 4 s of CPU (1 core); late started after 2 s and used 3 s.
        self.assertEqual(
            [(thread["name"], thread["cores"]) for thread in threads],
            [("main", 1.0), ("late", 0.75), ("worker", 0.502)],
        )
        self.assertIsNone(thread_usage(100.0, sweeps, 2.5, 3.5))

    def test_a_reused_thread_id_counts_from_zero(self) -> None:
        sweeps = [(0.0, {5: (1, 900, "old")}), (1.0, {5: (2, 50, "new")})]
        self.assertEqual(thread_usage(100.0, sweeps, 0.0, 1.0)[0]["cores"], 0.5)

    def test_metadata_lists_threads_per_side(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_samples(run / "c.log", cores=1.0)
            (run / "t.log").write_text(
                "hz 100\n0.0 sweep\n0.0 4 4 0 nixlbench\n5.0 sweep\n5.0 4 4 500 nixlbench\n",
                encoding="utf-8",
            )
            details = cpu_metadata(
                run, {"cpu_files": {"target": "c.log"}, "thread_files": {"target": "t.log"}}, 0.0, 5.0
            )
        self.assertEqual(details["cpu_threads_target"], [{"tid": 4, "pid": 4, "name": "nixlbench", "cores": 1.0}])
        self.assertNotIn("cpu_threads_initiator", details)


class CpuSamplerTests(unittest.TestCase):
    def test_sampler_stamps_lines_with_the_local_clock(self) -> None:
        lines = ["cpu 1000000 0 10 0", "noise", "cpu 2000000 500 20 1", "cpu 3000000 500 30 1"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpu-initiator.log"
            origin = time.monotonic()
            with CpuSampler(LineKubectl(lines), "ns", {"initiator": "pod"}, {"initiator": path}, origin):
                deadline = time.monotonic() + 5
                while len(read_samples(path)) < 3 and time.monotonic() < deadline:
                    time.sleep(0.02)
            samples = read_samples(path)
        self.assertEqual([sample[1] for sample in samples], [1.0, 2.0, 3.0])
        self.assertEqual(samples, sorted(samples))
        self.assertTrue(all(0 <= sample[0] < 5 for sample in samples))

    def test_thread_lines_go_to_their_own_file(self) -> None:
        lines = [
            "hz 100", "cpu 1000000 0 10 0", "sweep", "thr 7 7 100 nixlbench", "thr 9 7 50 hp tcp worker",
            "cpu 2000000 0 20 0", "sweep", "thr 7 7 300 nixlbench", "thr 9 7 60 hp tcp worker",
        ]
        with tempfile.TemporaryDirectory() as directory:
            cpu_path = Path(directory) / "cpu-target.log"
            thread_path = Path(directory) / "threads-target.log"
            with CpuSampler(
                LineKubectl(lines), "ns", {"target": "pod"}, {"target": cpu_path}, time.monotonic(),
                thread_files={"target": thread_path},
            ):
                deadline = time.monotonic() + 5
                while len(read_thread_sweeps(thread_path)[1]) < 2 and time.monotonic() < deadline:
                    time.sleep(0.02)
            hz, sweeps = read_thread_sweeps(thread_path)
            self.assertEqual(len(read_samples(cpu_path)), 2)
        self.assertEqual(hz, 100.0)
        self.assertEqual(len(sweeps), 2)
        self.assertEqual(sweeps[1][1], {7: (7, 300, "nixlbench"), 9: (7, 60, "hp tcp worker")})

    def test_a_pod_that_cannot_be_sampled_leaves_an_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpu-target.log"
            with CpuSampler(Mock(), "ns", {"target": "pod"}, {"target": path}, time.monotonic()):
                pass
            self.assertEqual(path.read_text(encoding="utf-8"), "")

    @unittest.skipUnless(
        shutil.which("sh") and Path("/sys/fs/cgroup/cpu.stat").is_file(),
        "needs sh and a cgroup v2 cpu.stat",
    )
    def test_sampler_script_prints_counters(self) -> None:
        result = subprocess.run(
            ["timeout", "0.3", "sh", "-c", SAMPLER_SCRIPT, "cpu-sampler", "0.05", "2"],
            capture_output=True,
            text=True,
        )
        lines = [line.split() for line in result.stdout.splitlines()]
        self.assertEqual(lines[0][0], "hz")
        cpu = [fields for fields in lines if fields[0] == "cpu"]
        self.assertGreaterEqual(len(cpu), 2)
        self.assertTrue(all(len(fields) == 5 for fields in cpu))
        self.assertLessEqual(int(cpu[0][1]), int(cpu[-1][1]))
        self.assertIn(["sweep"], lines)
        threads = [fields for fields in lines if fields[0] == "thr"]
        self.assertTrue(threads)
        self.assertTrue(all(len(fields) >= 5 and fields[3].isdigit() for fields in threads))

    @unittest.skipUnless(
        shutil.which("sh") and Path("/sys/fs/cgroup/cpu.stat").is_file(),
        "needs sh and a cgroup v2 cpu.stat",
    )
    def test_samplers_are_stopped_in_the_pod_not_just_locally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = f"{directory}/sampler-"
            sampler = subprocess.Popen(
                ["sh", "-c", SAMPLER_SCRIPT, "cpu-sampler", "0.05", "2", f"{prefix}1.pid"],
                stdout=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + 5
                while not Path(f"{prefix}1.pid").is_file() and time.monotonic() < deadline:
                    time.sleep(0.02)
                result = subprocess.run(stop_command(prefix), capture_output=True, text=True, timeout=10)
                self.assertEqual(result.stdout.split(), ["stopped", str(sampler.pid)])
                self.assertEqual(sampler.wait(timeout=5), -15)  # SIGTERM
                self.assertFalse(Path(f"{prefix}1.pid").exists())
                # Nothing left to stop.
                self.assertEqual(subprocess.run(stop_command(prefix), capture_output=True, text=True).stdout, "")
            finally:
                sampler.kill()

    def test_the_sampler_stops_its_pod_side_before_and_after(self) -> None:
        class Kubectl(LineKubectl):
            def __init__(self, lines):
                super().__init__(lines)
                self.runs = []

            def run(self, arguments, *, capture_output=False, timeout=None):
                self.runs.append(arguments)
                return subprocess.CompletedProcess(arguments, 0, "stopped 42\n", "")

        kubectl = Kubectl(["cpu 1000000 0 10 0"])
        with tempfile.TemporaryDirectory() as directory:
            sampler = CpuSampler(kubectl, "ns", {"target": "pod"}, {"target": Path(directory) / "cpu.log"},
                                 time.monotonic())
            with sampler:
                pass
        self.assertEqual(len(kubectl.runs), 2)
        self.assertEqual(kubectl.runs[0][:5], ["exec", "-n", "ns", "pod", "--"])
        self.assertEqual(kubectl.runs[0][5:], stop_command())
        self.assertEqual(sampler.cleanups, [{"target": "stopped 42"}, {"target": "stopped 42"}])

    def test_cleanup_is_confirmed_only_when_every_sampler_stopped(self) -> None:
        self.assertEqual(cleanup_problems({"target": "stopped 4\nstopped 9", "initiator": ""}), [])
        self.assertEqual(cleanup_problems({"target": "stopped 4\nrunning 9"}), ["target: stopped 4\nrunning 9"])
        self.assertEqual(cleanup_problems({"target": "error: pod gone"}), ["target: error: pod gone"])


NIXL_ROWS = """\
Num threads (--num_threads=N)                               : 2
Block Size (B)      Batch Size     B/W (GB/Sec)   Avg Lat. (us)
1024                1              2.0            1.0
1024                2              4.0            1.0
"""


def write_nixl_run(root: Path, *, cpu: bool = True) -> None:
    run = root / "kv" / "raw" / "t2"
    run.mkdir(parents=True)
    (run / "kv-t2-run1-initiator.log").write_text(NIXL_ROWS, encoding="utf-8")
    (run / "kv-t2-run1-target.log").write_text("ready\n", encoding="utf-8")
    command = {
        "backend": "nixl", "test_id": "kv", "run_number": 1, "parallelism": "threads",
        "initiator_file": "kv-t2-run1-initiator.log", "target_file": "kv-t2-run1-target.log",
        "initiator_status": 0, "target_status": 0,
        # 8 GB at 2 GB/s ends at 6 s, so it ran 2-6 s; 8 GB at 4 GB/s ran 10-12 s.
        "rows": [
            {"block_size": 1024, "batch_size": 1, "timed_bytes": 8e9, "seen_s": 6.0},
            {"block_size": 1024, "batch_size": 2, "timed_bytes": 8e9, "seen_s": 12.0},
        ],
    }
    if cpu:
        command["cpu_files"] = {"initiator": "kv-t2-run1-cpu-initiator.log", "target": "kv-t2-run1-cpu-target.log"}
        write_samples(run / "kv-t2-run1-cpu-initiator.log", cores=2.0)
        write_samples(run / "kv-t2-run1-cpu-target.log", cores=4.0)
    (run / "kv-t2-run1-command.json").write_text(json.dumps(command), encoding="utf-8")


class CpuParsingTests(unittest.TestCase):
    def test_nixl_rows_carry_the_cpu_of_their_timed_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_nixl_run(Path(directory))
            first, second = NixlParser().parse(Path(directory) / "kv" / "raw" / "t2")
        self.assertAlmostEqual(first.metadata["cpu_window_s"], 4.0)
        self.assertAlmostEqual(second.metadata["cpu_window_s"], 2.0)
        self.assertAlmostEqual(first.metadata["cpu_cores_initiator"], 2.0)
        self.assertAlmostEqual(first.metadata["cpu_cores_target"], 4.0)

    def test_nixl_runs_without_samples_have_no_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_nixl_run(Path(directory), cpu=False)
            observations = NixlParser().parse(Path(directory) / "kv" / "raw" / "t2")
        self.assertNotIn("cpu_cores_initiator", observations[0].metadata)

    def test_iperf3_intervals_carry_the_cpu_of_the_measured_part(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "t1"
            run.mkdir()
            write_samples(run / "cpu-i.log", cores=1.0)
            write_samples(run / "cpu-t.log", cores=1.5)
            (run / "r-command.json").write_text(json.dumps({
                "backend": "iperf3", "test_id": "r", "run_number": 1,
                "parameters": {"threads": 1, "duration_seconds": 5},
                "client_file": "r-client.json", "server_file": "../r-server.json",
                "client_status": 0, "client_finished_s": 12.0,
                "cpu_files": {"initiator": "cpu-i.log", "target": "cpu-t.log"},
            }), encoding="utf-8")
            (run / "r-client.json").write_text(json.dumps({
                "start": {"test_start": {"num_streams": 1}},
                "server_output_json": {"intervals": [
                    {"sum": {"start": 0, "end": 1, "seconds": 1, "bits_per_second": 1e9}},
                ]},
            }), encoding="utf-8")
            (Path(directory) / "r-server.json").write_text("server\n", encoding="utf-8")
            (observation,) = Iperf3Parser().parse(run)
        self.assertAlmostEqual(observation.metadata["cpu_cores_initiator"], 1.0)
        self.assertAlmostEqual(observation.metadata["cpu_cores_target"], 1.5)
        self.assertAlmostEqual(observation.metadata["cpu_window_s"], 5.0)


class CpuGraphTests(unittest.TestCase):
    def observation(self, **metadata: object) -> Observation:
        return Observation(
            metric="throughput", value=80e9, unit="bit/s", dimensions={"threads": 1},
            sample_kind="row", run_id="r1", metadata=metadata,
        )

    def test_each_side_becomes_its_own_observation(self) -> None:
        source = (self.observation(cpu_cores_initiator=2.0, cpu_cores_target=4.0), self.observation())
        cpu = cpu_observations(source, "cpu")
        per_gb = cpu_observations(source, "cpu-per-gb")
        self.assertEqual([o.dimensions["side"] for o in cpu], ["initiator", "target"])
        self.assertEqual([o.value / 1e9 for o in cpu], [2.0, 4.0])
        # 80 Gbit/s is 10 GB/s: 2 cores cost 0.2 core-seconds per GB, 4 cores 0.4.
        self.assertEqual([round(o.value / 1e9, 9) for o in per_gb], [0.2, 0.4])

    def test_cost_per_gb_uses_the_mean_rate_of_the_window(self) -> None:
        def interval(value: float, valid: bool = True) -> Observation:
            return Observation(
                metric="throughput", value=value, unit="bit/s", dimensions={"threads": 1},
                sample_kind="interval", run_id="r1",
                metadata={"valid": valid, "cpu_cores_initiator": 1.0, "cpu_window_s": 5.0},
            )
        # 40 and 120 Gbit/s average to 10 GB/s; the omitted interval is ignored.
        source = (interval(40e9), interval(120e9), interval(1e9, valid=False))
        values = [o.value / 1e9 for o in cpu_observations(source, "cpu-per-gb")]
        self.assertEqual([round(value, 9) for value in values], [0.1, 0.1, 0.1])

    def test_datasets_without_samples_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no CPU samples"):
            cpu_observations((self.observation(),), "cpu")

    def test_cli_plots_cpu_per_side(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_nixl_run(Path(directory))
            arguments = ["--backend", "nixl", "--name", "kv", "--dir", directory, "--x", "batch_size",
                         "--filter", "threads=2", "--filter", "block_size=1K"]
            self.assertEqual(graph_main([*arguments, "--y", "cpu"], default_registry()), 0)
            self.assertEqual(
                graph_main([*arguments, "--y", "cpu-per-gb", "--filter", "side=target"], default_registry()),
                0,
            )
            graphs = sorted(path.name for path in (Path(directory) / "kv" / "graphs").iterdir())
            self.assertEqual(
                graphs,
                [
                    "cpu-batch_size__block_size-1K__threads-2.png",
                    "cpu-per-gb-batch_size__block_size-1K__side-target__threads-2.png",
                ],
            )
            self.assertEqual(
                graph_main([*arguments, "--y", "cpu", "--max-throughput", "100"], default_registry()), 2
            )


if __name__ == "__main__":
    unittest.main()
