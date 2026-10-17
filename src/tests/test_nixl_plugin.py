from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.backends.nixl.analyzer import NixlAnalyzer
from src.backends.nixl.parser import NixlParser
from src.backends.nixl.plugin import create_plugin
from src.backends.nixl.runner import NixlRunner
from src.backends.registry import default_registry
from src.benchmark.contracts import BackendResources, BenchmarkConfig
from src.graphing.contracts import AnalysisRequest


NIXL_INITIATOR_FIXTURE = """\
Num threads (--num_threads=N)                               : 2
Block Size (B)      Batch Size     B/W (GB/Sec)   Avg Lat. (us)
4194304             1              1.575178
4194304             2              2.123026
4194304             4              2.349421
4194304             8              2.294085
8388608             1              2.301086
8388608             2              1.972920
8388608             4              2.034126
8388608             8              2.397880
16777216            1              2.096955
16777216            2              1.779816
16777216            4              2.431657
16777216            8              2.439225
33554432            1              2.410953
33554432            2              2.419340
33554432            4              2.445068
33554432            8              2.436507
67108864            1              2.425439
67108864            2              2.446416
67108864            4              2.445558
67108864            8              2.447944
"""
NIXL_TARGET_FIXTURE = """\
Using address 10.130.3.43 port 12345 for ASIO runtime
All processes are ready to proceed
"""


class NixlPluginTests(unittest.TestCase):
    def make_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            backend="nixl",
            test_id="nixl-test",
            initiator_node="worker-a",
            target_node="worker-b",
            namespace="bench",
            image="example/nixl:latest",
            log_directory=Path("logs"),
            parameters={"iterations": 100, "warmup_iterations": 10},
        )

    def make_fixture(self, directory: Path, threads: int = 2) -> Path:
        raw = directory / "raw"
        run = raw / f"t{threads}"
        run.mkdir(parents=True)
        (run / "nixl-test-t2-run1-initiator.log").write_text(
            NIXL_INITIATOR_FIXTURE,
            encoding="utf-8",
        )
        (run / "nixl-test-t2-run1-target.log").write_text(
            NIXL_TARGET_FIXTURE,
            encoding="utf-8",
        )
        command = {
            "backend": "nixl",
            "test_id": "nixl-test",
            "run_number": 1,
            "parameters": {"num_threads": threads},
            "initiator_file": "nixl-test-t2-run1-initiator.log",
            "target_file": "nixl-test-t2-run1-target.log",
            "initiator_status": 0,
            "target_status": 0,
        }
        (run / "nixl-test-t2-run1-command.json").write_text(
            json.dumps(command), encoding="utf-8"
        )
        return raw

    def test_plugin_is_registered(self) -> None:
        plugin = create_plugin()
        self.assertIsInstance(plugin.create_runner(), NixlRunner)
        self.assertIsInstance(plugin.create_analyzer(), NixlAnalyzer)
        self.assertIn("nixl", default_registry().names())
        self.assertEqual(
            plugin.graph_dimensions,
            frozenset({"threads", "block_size", "batch_size"}),
        )

    def test_parser_reads_twenty_table_rows(self) -> None:
        parser = NixlParser()
        with tempfile.TemporaryDirectory() as directory:
            raw = self.make_fixture(Path(directory))
            observations = parser.parse(parser.discover(raw)[0])
        self.assertEqual(len(observations), 20)
        self.assertEqual(observations[0].dimensions["threads"], 2)
        self.assertEqual(observations[0].dimensions["block_size_bytes"], 4194304)
        self.assertEqual(observations[0].dimensions["batch_size"], 1)
        self.assertAlmostEqual(observations[0].value, 1.575178e9 * 8)

    def make_processes(self, directory: Path, rates: list[list[float]], blocks=((4194304, 1), (4194304, 2)), timing=None) -> Path:
        """A processes run: rates[point][process] in GB/s for each (block, batch) point."""

        raw = directory / "raw"
        run = raw / f"t{len(rates[0])}"
        run.mkdir(parents=True)
        points = []
        for (block, batch), point_rates in zip(blocks, rates):
            files = []
            for index, rate in enumerate(point_rates, start=1):
                stem = f"nixl-test-b{block}-n{batch}-p{index}"
                (run / f"{stem}-initiator.log").write_text(
                    "Block Size (B)      Batch Size     B/W (GB/Sec)   Avg Lat. (us)\n"
                    f"{block:<20}{batch:<15}{rate:<15.6f}66.0\n",
                    encoding="utf-8",
                )
                (run / f"{stem}-target.log").write_text(NIXL_TARGET_FIXTURE, encoding="utf-8")
                files.append({
                    "asio_port": 12345, "initiator_file": f"{stem}-initiator.log",
                    "target_file": f"{stem}-target.log",
                })
            point = {"block_size": block, "batch_size": batch, "files": files}
            if timing is not None:
                timed_bytes, seen = timing[len(points)]
                point["timed_bytes"] = timed_bytes
                for entry, value in zip(files, seen):
                    entry["row_seen_s"] = value
            points.append(point)
        command = {
            "backend": "nixl", "test_id": "nixl-test", "run_number": 1,
            "parallelism": "processes", "processes": len(rates[0]), "points": points,
            "initiator_status": 0, "target_status": 0,
        }
        (run / "nixl-test-t2-run1-command.json").write_text(json.dumps(command), encoding="utf-8")
        return raw

    def test_processes_aggregate_is_n_times_the_slowest_process(self) -> None:
        parser = NixlParser()
        with tempfile.TemporaryDirectory() as directory:
            raw = self.make_processes(Path(directory), [[4.0, 4.0, 2.3, 2.3], [3.0, 3.0, 3.0, 3.0]])
            observations = parser.parse(parser.discover(raw)[0])
        self.assertEqual(len(observations), 2)
        first, second = observations
        self.assertEqual(first.dimensions, {"threads": 4, "block_size_bytes": 4194304, "batch_size": 1})
        # Adding the averages would give 12.6 GB/s; the aggregate is 4 x 2.3 GB/s.
        self.assertAlmostEqual(first.value, 4 * 2.3 * 8e9)
        self.assertAlmostEqual(first.metadata["skew"], 4.0 / 2.3)
        self.assertEqual(first.metadata["process_rates_GB_s"], [4.0, 4.0, 2.3, 2.3])
        self.assertAlmostEqual(second.value, 12.0 * 8e9)
        self.assertAlmostEqual(second.metadata["skew"], 1.0)
        self.assertEqual(len(first.metadata["initiator_files"]), 4)

    def test_aligned_processes_add_up_over_the_measured_window(self) -> None:
        parser = NixlParser()
        timing = [(10e9, [10.0, 10.0]), (10e9, [10.0, 10.0])]
        with tempfile.TemporaryDirectory() as directory:
            raw = self.make_processes(Path(directory), [[1.0, 1.0], [1.0, 1.0]], timing=timing)
            observations = parser.parse(parser.discover(raw)[0])
        first = observations[0]
        self.assertAlmostEqual(first.value, 2.0 * 8e9)
        self.assertEqual(first.metadata["aggregate_method"], "window")
        self.assertAlmostEqual(first.metadata["window_s"], 10.0)
        self.assertAlmostEqual(first.metadata["overlap"], 1.0)

    def test_staggered_processes_are_averaged_over_the_whole_window(self) -> None:
        parser = NixlParser()
        # Both ran 10 s, but the second started 2 s later: they were together for 8 s of 12.
        timing = [(10e9, [10.0, 12.0]), (10e9, [10.0, 10.0])]
        with tempfile.TemporaryDirectory() as directory:
            raw = self.make_processes(Path(directory), [[1.0, 1.0], [1.0, 1.0]], timing=timing)
            first = parser.parse(parser.discover(raw)[0])[0]
        # 20 GB over 12 s, not the 2 GB/s the sum of the rates would claim.
        self.assertAlmostEqual(first.value, 20.0 / 12.0 * 8e9)
        self.assertAlmostEqual(first.metadata["window_s"], 12.0)
        self.assertAlmostEqual(first.metadata["overlap"], 8.0 / 12.0)

    def test_window_aggregate_never_exceeds_n_times_the_slowest(self) -> None:
        parser = NixlParser()
        timing = [(10e9, [4.0, 5.0, 4.5])]
        with tempfile.TemporaryDirectory() as directory:
            raw = self.make_processes(
                Path(directory), [[3.0, 2.0, 2.5]], blocks=((4194304, 1),), timing=timing
            )
            (observation,) = parser.parse(parser.discover(raw)[0])
        self.assertLessEqual(observation.value, 3 * 2.0 * 8e9 + 1)
        self.assertEqual(observation.metadata["aggregate_method"], "window")

    def test_missing_timing_falls_back_to_the_slowest_rate(self) -> None:
        parser = NixlParser()
        timing = [(10e9, [10.0, None]), (10e9, [10.0, 10.0])]
        with tempfile.TemporaryDirectory() as directory:
            raw = self.make_processes(Path(directory), [[1.0, 1.5], [1.0, 1.0]], timing=timing)
            first, second = parser.parse(parser.discover(raw)[0])
        self.assertEqual(first.metadata["aggregate_method"], "slowest_rate")
        self.assertAlmostEqual(first.value, 2.0 * 1.0 * 8e9)
        self.assertEqual(second.metadata["aggregate_method"], "window")

    def test_processes_aggregate_cannot_exceed_the_slowest_times_n(self) -> None:
        parser = NixlParser()
        with tempfile.TemporaryDirectory() as directory:
            raw = self.make_processes(Path(directory), [[4.0, 4.0, 4.8, 4.7]])
            (value,) = [o.value for o in parser.parse(parser.discover(raw)[0])]
        self.assertAlmostEqual(value, 4 * 4.0 * 8e9)

    def test_processes_run_with_a_missing_row_is_rejected(self) -> None:
        parser = NixlParser()
        with tempfile.TemporaryDirectory() as directory:
            raw = self.make_processes(Path(directory), [[3.0, 3.0]])
            (raw / "t2" / "nixl-test-b4194304-n1-p2-initiator.log").write_text("no rows\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected one block_size=4194304 batch_size=1 row"):
                parser.parse(parser.discover(raw)[0])

    def test_datasets_from_the_row_summing_version_are_rejected(self) -> None:
        parser = NixlParser()
        with tempfile.TemporaryDirectory() as directory:
            raw = self.make_fixture(Path(directory))
            command_path = raw / "t2" / "nixl-test-t2-run1-command.json"
            command = json.loads(command_path.read_text(encoding="utf-8"))
            command["instance_files"] = [{"initiator_file": "a", "target_file": "b"}]
            command_path.write_text(json.dumps(command), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run it again"):
                parser.parse(parser.discover(raw)[0])

    def test_analyzer_requires_filter_or_series_for_varying_dimensions(self) -> None:
        parser = NixlParser()
        analyzer = NixlAnalyzer(parser)
        with tempfile.TemporaryDirectory() as directory:
            raw = self.make_fixture(Path(directory))
            observations = analyzer.parse(analyzer.discover(raw)[0])
            with self.assertRaises(ValueError):
                analyzer.aggregate(
                    observations,
                    AnalysisRequest(metric="throughput", x_dimension="block_size_bytes"),
                )

    def test_repetitions_give_a_mean_and_a_standard_deviation(self) -> None:
        analyzer = NixlAnalyzer(NixlParser())
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            run = raw / "t2"
            run.mkdir(parents=True)
            for number, scale in ((1, 1.0), (2, 1.1), (3, 0.9)):
                table = "\n".join(
                    line if not line.strip()[:1].isdigit() or line.startswith("Num")
                    else line.replace(line.split()[2], f"{float(line.split()[2]) * scale:.6f}")
                    for line in NIXL_INITIATOR_FIXTURE.splitlines()
                ) + "\n"
                stem = f"nixl-test-t2-run{number}"
                (run / f"{stem}-initiator.log").write_text(table, encoding="utf-8")
                (run / f"{stem}-target.log").write_text(NIXL_TARGET_FIXTURE, encoding="utf-8")
                (run / f"{stem}-command.json").write_text(json.dumps({
                    "backend": "nixl", "test_id": "nixl-test", "run_number": number,
                    "parameters": {"num_threads": 2},
                    "initiator_file": f"{stem}-initiator.log", "target_file": f"{stem}-target.log",
                    "initiator_status": 0, "target_status": 0,
                }), encoding="utf-8")
            observations = analyzer.parse(analyzer.discover(raw)[0])
            points = analyzer.aggregate(
                observations,
                AnalysisRequest(
                    metric="throughput", x_dimension="block_size_bytes",
                    filters={"batch_size": 1},
                ),
            )
        self.assertEqual(len(observations), 60)
        first = points[0]
        base = 1.575178 * 8
        self.assertAlmostEqual(first.y_value, base, places=4)  # mean of 0.9, 1.0 and 1.1 times the base
        self.assertAlmostEqual(first.y_error[0], base * 0.1, places=4)  # sample standard deviation
        self.assertEqual(first.y_error[0], first.y_error[1])

    def test_a_single_run_has_no_spread(self) -> None:
        analyzer = NixlAnalyzer(NixlParser())
        with tempfile.TemporaryDirectory() as directory:
            raw = self.make_fixture(Path(directory))
            observations = analyzer.parse(analyzer.discover(raw)[0])
            points = analyzer.aggregate(
                observations,
                AnalysisRequest(
                    metric="throughput", x_dimension="block_size_bytes",
                    filters={"batch_size": 1},
                ),
            )
        self.assertEqual(points[0].y_error, (0.0, 0.0))

    def test_analyzer_filters_batch_size(self) -> None:
        parser = NixlParser()
        analyzer = NixlAnalyzer(parser)
        with tempfile.TemporaryDirectory() as directory:
            raw = self.make_fixture(Path(directory))
            observations = analyzer.parse(analyzer.discover(raw)[0])
            points = analyzer.aggregate(
                observations,
                AnalysisRequest(
                    metric="throughput",
                    x_dimension="block_size_bytes",
                    filters={"batch_size": 8},
                ),
            )
        self.assertEqual(len(points), 5)
        self.assertEqual(points[0].x_value, 4194304)
        self.assertEqual(points[0].y_value, 2.294085 * 8)

    def test_runner_passes_calculated_buffer_and_runtime_arguments(self) -> None:
        runner = NixlRunner()
        args = runner._nixl_args(2, 50, 10, 1073741824, 4194304, 67108864, 1, 8, "10.0.0.1")
        self.assertEqual(
            args,
            [
                "--runtime_type", "ASIO",
                "--asio_address", "10.0.0.1",
                "--asio_port", "12345",
                "--backend", "UCX",
                "--initiator_seg_type", "DRAM",
                "--target_seg_type", "DRAM",
                "--total_buffer_size", "1073741824",
                "--start_block_size", "4194304",
                "--max_block_size", "67108864",
                "--start_batch_size", "1",
                "--max_batch_size", "8",
                "--num_threads", "2",
                "--num_iter", "50",
                "--warmup_iter", "10",
            ],
        )
    def test_runner_renders_manifest_with_role_mapping(self) -> None:
        runner = NixlRunner()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "manifest.yaml"
            runner.render_manifest(self.make_config(), destination)
            text = destination.read_text(encoding="utf-8")
        self.assertIn("nodeName: worker-a", text)
        self.assertIn("nodeName: worker-b", text)
        self.assertNotIn("__TEST_ID__", text)


if __name__ == "__main__":
    unittest.main()
