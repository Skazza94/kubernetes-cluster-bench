from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

from src.backends.registry import default_registry
from src.backends.vllm.analyzer import VllmAnalyzer
from src.backends.vllm.parser import VllmParser, summarize, verify
from src.backends.vllm.runner import VllmRunner
from src.benchmark.contracts import BackendResources, BenchmarkConfig
from src.benchmark.network import NetworkAttachment, SecondaryNetwork
from src.graphing.cli import main as graph_main
from src.graphing.contracts import AnalysisRequest


IDS = {64: "11111111-1111-4111-8111-111111111111", 96: "22222222-2222-4222-8222-222222222222",
       33: "33333333-3333-4333-8333-333333333333"}


def pid(length: int) -> str:
    return f"cmpl-{IDS[length]}-0-aaaaaaaa"


def did(length: int) -> str:
    return f"cmpl-{IDS[length]}-0-bbbbbbbb"


def step_log(*requests: tuple) -> str:
    """One simulator step line per (req_id, prompt tokens, computed, scheduled[,
    t_start, t_deadline])."""

    lines = []
    for req_id, prompt, computed, scheduled, *times in requests:
        line = {"step": {"requests": [{
            "req_id": req_id, "num_prompt_tokens": prompt,
            "num_computed_tokens": computed, "num_scheduled_tokens": scheduled,
        }]}}
        if times:
            line["t_start"], line["t_deadline"] = times
        lines.append(json.dumps(line) + "\n")
    return "".join(lines)


def kv_line(length: int, t: float, ok: bool = True, block_ids=(1, 2)) -> str:
    check = {"req_id": did(length), "remote_request_id": pid(length), "blocks": 2,
             "layers": 2, "bytes": 262144, "mismatched_bytes": 0 if ok else 9, "ok": ok,
             "block_ids": list(block_ids)}
    return json.dumps({"kv_check": check, "t": t}) + "\n"


def events(t: float, **kinds) -> str:
    """A simulator kv_events line (real KV transfers)."""

    return json.dumps({"kv_events": kinds, "t": t}) + "\n"


def record(length: int, elapsed: float = 0.2, tokens=(70, 71, 72, 73), **extra) -> dict:
    return {
        "index": 0, "request_id": IDS[length], "prompt_length": length,
        "expected_token_ids": [70, 71, 72, 73],
        "t_sent": 10.0, "t_received": 10.0 + elapsed, "elapsed_s": elapsed,
        "response": {"id": f"cmpl-{IDS[length]}", "usage": {"prompt_tokens": length},
                     "choices": [{"token_ids": list(tokens), "finish_reason": "length"}]},
        **extra,
    }


def timed_out(length: int, **extra) -> dict:
    """A request the client gave up on."""

    lost = dict(record(length), error="TimeoutError: timed out", timed_out=True, **extra)
    lost.pop("response")
    return lost


GOOD_PREFILL = step_log((pid(64), 64, 0, 64, 1.0, 2.0), (pid(96), 96, 0, 96, 2.0, 3.0),
                        (pid(64), 64, 64, 1, 3.0, 3.1))
GOOD_DECODE = step_log((did(64), 64, 63, 1, 5.0, 5.1), (did(96), 96, 95, 1, 6.0, 6.1),
                       (did(64), 64, 64, 1, 7.0, 7.1))
CHECKED_DECODE = kv_line(64, 4.9, block_ids=(1, 2)) + kv_line(96, 5.9, block_ids=(2, 3)) + GOOD_DECODE
# Real mode: both servers' lifecycle events for the two transfers above.
REAL_PREFILL = (events(1.0, allocated={pid(64): [1, 2], pid(96): [3, 4, 5]}) + GOOD_PREFILL
                + events(3.0, finished=[pid(64), pid(96)])
                + events(4.95, sends_done=[pid(64)]) + events(5.95, sends_done=[pid(96)]))
REAL_DECODE = (events(4.8, allocated={did(64): [1, 2], did(96): [3, 4, 5]},
                      reads_posted=[did(64), did(96)])
               + events(4.9, reads_done=[did(64)]) + events(5.9, reads_done=[did(96)])
               + GOOD_DECODE + events(8.0, finished=[did(64), did(96)]))
METRICS = ('# HELP x\nvllm:nixl_num_failed_transfers_total{engine="0"} 0\n'
           'vllm:nixl_bytes_transferred_sum{engine="0"} 600000\n'
           'vllm:kv_cache_usage_perc{engine="0"} 0.0\n')


class FakeKubectl:
    """Answers the runner's kubectl exec calls like the pods would."""

    def __init__(self, decode_steps: str = GOOD_DECODE, metrics: str = METRICS,
                 connections=("10.0.0.2:40000->10.0.0.1:41000",)) -> None:
        self.decode_steps = decode_steps
        self.metrics = metrics
        self.connections = list(connections)
        self.calls: list[list[str]] = []

    def run(self, arguments, *, capture_output=False, suppress_stdout=False, timeout=None):
        self.calls.append(list(arguments))
        command = list(arguments[arguments.index("--") + 1:])
        pod = arguments[arguments.index("--") - 1]
        helper = command[:2] == ["/opt/venv/bin/python", "/bench/pod_helper.py"]
        if command[0] == "sh":
            stdout = "0\n"
        elif helper and command[2] == "scenario":
            lengths = command[command.index("--prompt-lengths") + 1].split(",")
            stdout = json.dumps({"scenario": command[3], "concurrency": 1,
                                 "requests": [record(int(n)) for n in lengths]})
        elif helper and command[2] == "fetch":
            stdout = self.metrics
        elif helper and command[2] == "connections":
            stdout = json.dumps(self.connections)
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
            ({"scenario": "nope"}, "scenario must be one of"),
            ({"scenario": "cancel"}, "needs kv_transfer_mode real"),
            ({"kv_transfer_mode": "real", "scenario": "cancel", "cancel_delays": "0,1"}, "positive seconds"),
            ({"kv_transfer_mode": "real", "scenario": "producer-failure", "threads_spec": "1,2"}, "one run"),
            ({"repeat": 0}, "positive integer"),
            ({"prompt_lengths": "200"}, "between 2 and 128"),
            ({"output_tokens": 0}, "positive integer"),
        ):
            with self.subTest(parameters=parameters), self.assertRaisesRegex(ValueError, message):
                runner.validate(self.config(**parameters))
        with self.assertRaisesRegex(ValueError, "vllm-sim/setup.sh"):
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

    def test_real_mode_selects_mooncake_hp_tcp_on_the_side_address(self) -> None:
        runner = self.runner()
        command = runner.server_command(self.config(kv_transfer_mode="real"), "10.0.0.2", "initiator")
        env = dict(item.split("=", 1) for item in command[1:command.index("vllm")])
        kv_transfer = json.loads(command[command.index("--kv-transfer-config") + 1])
        self.assertEqual(kv_transfer["kv_connector_extra_config"], {"backends": ["Mooncake"]})
        self.assertEqual(env["MC_USE_TENT"], "1")
        self.assertEqual(env["NIXL_MOONCAKE_IP_ADDR"], "10.0.0.2")
        self.assertNotIn("MC_FORCE_TCP", env)
        self.assertEqual(json.loads(env["MC_TENT_CONF"])["rpc_server_hostname"], "10.0.0.2")
        transports = json.loads(env["MC_TENT_CONF"])["transports"]
        self.assertEqual(transports["hp_tcp"], {"enable": True, "bind_address": "10.0.0.2",
                                                "advertise_address": "10.0.0.2",
                                                "rail_addresses": ["10.0.0.2"]})
        self.assertEqual({name for name, value in transports.items() if not value["enable"]},
                         {"tcp", "rdma", "shm", "gds", "ascend_direct"})
        simulated = runner.server_command(self.config(), "10.0.0.2", "initiator")
        self.assertFalse(any(item.startswith("MC_") for item in simulated))
        with self.assertRaisesRegex(ValueError, "MC_FORCE_TCP"):
            runner.validate(self.config(kv_transfer_mode="real", env={"MC_FORCE_TCP": 1}))

    def test_profile_env_overrides_the_transport_settings(self) -> None:
        override = self.config(kv_transfer_mode="real", env={"MC_TENT_CONF": '{"transports":{}}'})
        command = self.runner().server_command(override, "10.0.0.1", "target")
        self.assertIn('MC_TENT_CONF={"transports":{}}', command)
        self.assertEqual(sum(item.startswith("MC_TENT_CONF=") for item in command), 1)

    def test_pattern_options_go_to_their_side(self) -> None:
        runner = self.runner()
        config = self.config(kv_transfer_mode="real", kv_pattern_fill=True, kv_pattern_check=True)
        def simulation(role):
            command = runner.server_command(config, "10.0.0.1", role)
            return json.loads(command[command.index("--simulation-config") + 1])
        self.assertTrue(simulation("target")["kv_pattern_fill"])
        self.assertNotIn("kv_pattern_check", simulation("target"))
        self.assertTrue(simulation("initiator")["kv_pattern_check"])
        self.assertNotIn("kv_pattern_fill", simulation("initiator"))
        with self.assertRaisesRegex(ValueError, "needs kv_pattern_fill"):
            runner.validate(self.config(kv_transfer_mode="real", kv_pattern_check=True))
        with self.assertRaisesRegex(ValueError, "needs kv_transfer_mode real"):
            runner.validate(self.config(kv_pattern_fill=True))

    def test_verify_requires_matching_kv_bytes_when_checked(self) -> None:
        (check, _) = verify([record(64), record(96)], GOOD_PREFILL, CHECKED_DECODE, kv_check=True)
        self.assertEqual(check["kv_check"]["bytes"], 262144)
        with self.assertRaisesRegex(ValueError, "no KV byte check"):
            verify([record(64)], GOOD_PREFILL, GOOD_DECODE, kv_check=True)
        bad = kv_line(64, 4.9, ok=False) + GOOD_DECODE
        with self.assertRaisesRegex(ValueError, "differ from prefill's"):
            verify([record(64)], GOOD_PREFILL, bad, kv_check=True)
        # Without the option, byte checks are not required.
        verify([record(64)], GOOD_PREFILL, GOOD_DECODE)

    def test_decode_must_resume_after_its_read_completed(self) -> None:
        # The read completed (kv_check at t=5.5) after decode's first step began (5.0).
        late = kv_line(64, 5.5) + GOOD_DECODE
        with self.assertRaisesRegex(ValueError, "resumed before its KV read completed"):
            verify([record(64)], GOOD_PREFILL, late, kv_check=True)

    def test_prefill_must_not_hand_over_before_its_modeled_compute(self) -> None:
        # Prefill's 64-token step ends at t=2.0 on its clock (the helper's too).
        on_time = record(64, prefill={"t_sent": 0.5, "t_received": 2.05})
        (check,) = verify([on_time], GOOD_PREFILL, GOOD_DECODE, scenario="handoff")
        self.assertAlmostEqual(check["prefill_response_after_compute_s"], 0.05)
        early = record(64, prefill={"t_sent": 0.5, "t_received": 1.5})
        with self.assertRaisesRegex(ValueError, "before its modeled computation ended"):
            verify([early], GOOD_PREFILL, GOOD_DECODE, scenario="handoff")

    def test_tcp_connections_decodes_proc_net_tcp(self) -> None:
        from unittest import mock

        from src.backends.vllm import pod_helper

        table = (
            "  sl  local_address rem_address   st\n"
            "   0: 2400000A:3D17 2300000A:1F40 01 rest\n"   # 10.0.0.36:15639 -> 10.0.0.35:8000
            "   1: 0100007F:1F90 00000000:0000 0A rest\n"   # listening: skipped
        )
        with mock.patch.object(pod_helper.Path, "read_text", return_value=table):
            self.assertEqual(pod_helper.tcp_connections(), ["10.0.0.36:15639->10.0.0.35:8000"])

    def test_outcomes_and_what_each_scenario_accepts(self) -> None:
        checks = verify([record(64), record(96)], GOOD_PREFILL, GOOD_DECODE)
        self.assertEqual([c["outcome"] for c in checks], ["transferred", "transferred"])
        self.assertEqual([c["decode_cached_tokens"] for c in checks], [63, 95])
        recomputed = step_log((did(64), 64, 0, 64, 5.0, 5.1))
        with self.assertRaisesRegex(ValueError, "was fallback"):
            verify([record(64)], GOOD_PREFILL, recomputed)
        with self.assertRaisesRegex(ValueError, "was failed"):
            verify([record(64, tokens=(1, 2, 3, 4))], GOOD_PREFILL, GOOD_DECODE)
        with self.assertRaisesRegex(ValueError, "was failed"):
            verify([record(64)], GOOD_PREFILL, "")
        # After a warm-up transfer, a dead producer must show as a failure or a
        # recomputation, never a transfer (real mode also needs decode's events).
        killed = dict(record(96), error="HTTP 500: kv load failed", producer_killed=[7, 8])
        killed.pop("response")
        checks = verify([record(64), killed], GOOD_PREFILL, GOOD_DECODE, scenario="producer-failure")
        self.assertEqual([c["outcome"] for c in checks], ["transferred", "failed"])
        recomputed_96 = step_log((did(64), 64, 63, 1, 5.0, 5.1), (did(96), 96, 0, 96, 6.0, 6.1))
        checks = verify([record(64), record(96, producer_killed=[7])], GOOD_PREFILL, recomputed_96,
                        scenario="producer-failure")
        self.assertEqual(checks[1]["outcome"], "fallback")
        with self.assertRaisesRegex(ValueError, "expects failed or fallback"):
            verify([record(64), record(96, producer_killed=[7])], GOOD_PREFILL, GOOD_DECODE,
                   scenario="producer-failure")
        with self.assertRaisesRegex(ValueError, "needs a transfer before"):
            verify([killed], GOOD_PREFILL, GOOD_DECODE, scenario="producer-failure")
        # Requests the client gave up on are reported, never counted as transfers.
        checks = verify([timed_out(96, cancel_after_s=0.01), record(64)], GOOD_PREFILL, GOOD_DECODE,
                        scenario="cancel")
        self.assertEqual([c["outcome"] for c in checks], ["cancelled", "transferred"])
        with self.assertRaisesRegex(ValueError, "cancelled no request"):
            verify([record(64)], GOOD_PREFILL, GOOD_DECODE, scenario="cancel")

    def test_a_cancel_that_completed_is_checked_like_any_request(self) -> None:
        # The client meant to cancel after 1 s, but the response came first.
        completed = record(96, cancel_after_s=1.0)
        (check, _) = verify([completed, record(64)], GOOD_PREFILL, GOOD_DECODE)
        self.assertEqual(check["outcome"], "transferred")
        self.assertTrue(check["completed_before_cancel"])
        with self.assertRaisesRegex(ValueError, "cancelled no request"):
            verify([completed, record(64)], GOOD_PREFILL, GOOD_DECODE, scenario="cancel")
        recomputed = step_log((did(64), 64, 63, 1, 5.0, 5.1), (did(96), 96, 0, 96, 6.0, 6.1))
        with self.assertRaisesRegex(ValueError, "was fallback"):
            verify([completed, record(64)], GOOD_PREFILL, recomputed, scenario="cancel")

    def test_real_mode_needs_both_servers_lifecycle(self) -> None:
        checks = verify([record(64), record(96)], REAL_PREFILL, REAL_DECODE, real=True)
        self.assertEqual([c["outcome"] for c in checks], ["transferred", "transferred"])
        unreleased = REAL_PREFILL.replace(events(5.95, sends_done=[pid(96)]), "")
        with self.assertRaisesRegex(ValueError, "prefill never released"):
            verify([record(64), record(96)], unreleased, REAL_DECODE, real=True)
        unread = REAL_DECODE.replace(events(5.9, reads_done=[did(96)]), "")
        with self.assertRaisesRegex(ValueError, "no completed READ"):
            verify([record(64), record(96)], REAL_PREFILL, unread, real=True)
        # Decode gave block 2 to the 96-token request before the 64-token one ended.
        early = REAL_DECODE.replace("[3, 4, 5]", "[2, 4, 5]")
        with self.assertRaisesRegex(ValueError, "gave block 2 to request .* still held it"):
            verify([record(64), record(96)], REAL_PREFILL, early, real=True)
        summary = summarize(checks, METRICS, METRICS, REAL_PREFILL, REAL_DECODE)
        self.assertEqual((summary["prefill_block_handovers"], summary["decode_block_handovers"]), (0, 0))

    def test_cancellation_stages_on_decodes_clock(self) -> None:
        paused = timed_out(33, cancel_after_s=1.0,
                           producer_paused={"decode_kv_usage_before_resume": 0.05})
        prefill = (REAL_PREFILL + events(9.0, allocated={pid(33): [6]}) + events(9.1, finished=[pid(33)])
                   + events(9.6, sends_done=[pid(33)]))
        # Decode posts the READ (9.2), the abort lands (9.3), the READ ends (9.5).
        decode = (REAL_DECODE + events(9.2, allocated={did(33): [6]}, reads_posted=[did(33)])
                  + events(9.3, finished=[did(33)]) + events(9.5, reads_done=[did(33)]))
        records = [paused, record(64), record(96)]
        (check, _, _) = verify(records, prefill, decode, scenario="cancel", real=True)
        self.assertEqual((check["outcome"], check["stage"], check["read"]),
                         ("cancelled", "READ outstanding", "done"))
        # The READ ended before the abort: the request was decoding.
        decoding = decode.replace(events(9.5, reads_done=[did(33)]), "").replace(
            events(9.3, finished=[did(33)]), events(9.25, reads_done=[did(33)]) + events(9.3, finished=[did(33)]))
        with self.assertRaisesRegex(ValueError, "landed decoding"):
            verify(records, prefill, decoding, scenario="cancel", real=True)
        unpaused = timed_out(33, cancel_after_s=1.0)
        (check, _, _) = verify([unpaused, record(64), record(96)], prefill, REAL_DECODE + events(
            9.2, allocated={did(33): [6]}, reads_posted=[did(33)]) + events(9.25, reads_done=[did(33)])
            + events(9.3, finished=[did(33)]), scenario="cancel")
        with self.assertRaisesRegex(ValueError, "while its READ was outstanding"):
            verify([unpaused, record(64), record(96)], prefill, decoding, scenario="cancel", real=True)
        # The abort taken only in the step that saw the READ end (decode stuck).
        stuck = decode.replace(events(9.3, finished=[did(33)]) + events(9.5, reads_done=[did(33)]),
                               events(9.5, finished=[did(33)], reads_done=[did(33)]))
        (check, _, _) = verify(records, prefill, stuck, scenario="cancel", real=True)
        self.assertEqual(check["stage"], "abort taken when the READ ended")
        # The same step, but the abort was scheduled at its start (9.3): the
        # step then blocked until the READ ended.
        blocked = stuck.replace('"t": 9.5}', '"t_scheduled": 9.3, "t": 9.5}')
        (check, _, _) = verify(records, prefill, blocked, scenario="cancel", real=True)
        self.assertEqual(check["stage"], "READ outstanding")
        # A bystander decode computes itself: served only after the READ ended.
        bystander = dict(record(64), request_id=IDS[96], index=50, t_sent=11.2, t_received=12.0)
        by_decode = stuck + step_log((did(96), 64, 0, 64, 9.7, 9.8))
        watched = dict(paused, bystanders=[bystander], t_received=11.0,
                       producer_paused={"decode_kv_usage_before_resume": 0.05, "t_resumed": 11.5})
        (check,) = verify([watched], prefill, by_decode.replace(GOOD_DECODE, ""), real=True)
        (seen,) = check["bystanders"]
        self.assertAlmostEqual(seen["first_step_after_read_end_s"], 0.2)
        self.assertAlmostEqual(seen["answered_after_resume_s"], 0.5)
        self.assertAlmostEqual(seen["sent_after_cancel_s"], 0.2)
        lost = dict(watched, bystanders=[timed_out(96)])
        with self.assertRaisesRegex(ValueError, "did not serve a bystander"):
            verify([lost], prefill, stuck, real=True)
        # vLLM's serving layer also tells prefill to free blocks decode already read.
        twice = decode + events(9.31, prefill_notified=[pid(33)])
        (check, _, _) = verify(records, prefill, twice, scenario="cancel", real=True)
        self.assertTrue(check["duplicate_free_notification"])
        idle = dict(paused, producer_paused={"decode_kv_usage_before_resume": 0.0})
        with self.assertRaisesRegex(ValueError, "held no KV blocks"):
            verify([idle, record(64), record(96)], prefill, decode, scenario="cancel", real=True)
        # Never admitted by decode: prefill must still let go (lease expiry).
        (check, _, _) = verify([unpaused, record(64), record(96)], prefill, REAL_DECODE, scenario="cancel")
        self.assertEqual(check["outcome"], "cancelled")
        notified = REAL_DECODE + events(9.2, prefill_notified=[pid(33)])
        checks = verify([unpaused, record(64), record(96)], prefill, notified, real=True)
        self.assertEqual(checks[0]["stage"], "before decode scheduled it")
        unreleased = prefill.replace(events(9.6, sends_done=[pid(33)]), "")
        with self.assertRaisesRegex(ValueError, "prefill never released"):
            verify([dict(unpaused), record(64), record(96)], unreleased, REAL_DECODE + events(
                9.2, allocated={did(33): [6]}, reads_posted=[did(33)]) + events(9.3, finished=[did(33)])
                + events(9.5, reads_done=[did(33)]), scenario="cancel", real=True)
        with self.assertRaisesRegex(ValueError, "decode never ended the request"):
            verify([unpaused, record(64), record(96)], prefill, REAL_DECODE + events(
                9.2, allocated={did(33): [6]}, reads_posted=[did(33)]), scenario="cancel", real=True)

    def test_a_killed_producer_needs_a_failed_read_and_an_answer(self) -> None:
        killed = dict(timed_out(96), error="HTTP 500: Internal server error", producer_killed=[7])
        killed.pop("timed_out")
        prefill = REAL_PREFILL.replace(events(5.95, sends_done=[pid(96)]), "")
        failed = REAL_DECODE.replace(events(5.9, reads_done=[did(96)]), events(5.9, reads_failed=[did(96)]))
        checks = verify([record(64), killed], prefill, failed, scenario="producer-failure", real=True)
        self.assertEqual(checks[1]["outcome"], "failed")
        with self.assertRaisesRegex(ValueError, "no failed READ"):
            verify([record(64), killed], prefill, REAL_DECODE.replace(events(5.9, reads_done=[did(96)]), ""),
                   scenario="producer-failure", real=True)
        silent = dict(timed_out(96), producer_killed=[7])
        with self.assertRaisesRegex(ValueError, "client timed out"):
            verify([record(64), silent], prefill, failed, scenario="producer-failure", real=True)
        # An HTTP 200 with wrong tokens, or one decode never ran, is no
        # producer failure, whatever the events say.
        wrong = record(96, tokens=(0, 0, 0, 0), producer_killed=[7])
        with self.assertRaisesRegex(ValueError, "answered successfully with a wrong response"):
            verify([record(64), wrong], prefill, failed, scenario="producer-failure", real=True)
        no_events = REAL_DECODE.replace(events(5.9, reads_done=[did(96)]), "")
        with self.assertRaisesRegex(ValueError, "wrong response.*no failed READ"):
            verify([record(64), wrong], prefill, no_events, scenario="producer-failure", real=True)
        unrun = REAL_DECODE.replace(step_log((did(96), 96, 95, 1, 6.0, 6.1)), "").replace(
            events(5.9, reads_done=[did(96)]), "")
        with self.assertRaisesRegex(ValueError, "never ran on decode.*no failed READ"):
            verify([record(64), dict(record(96), producer_killed=[7])], prefill, unrun,
                   scenario="producer-failure", real=True)

    def test_pressure_reuses_held_blocks_only_after_their_release(self) -> None:
        # Held: 64 (blocks 1-2) and 96 (3-5). 33 waits, then gets block 2 once
        # decode has read the 64-token request (sent at t=10.0, released 10.5).
        held = [record(64, held=True), record(96, held=True)]
        held[0]["t_sent"] = 10.0
        held[1]["t_sent"] = 11.0
        pressure = {"still_queued": True, "prefill_waiting": 1.0}
        blocked = dict(record(64), request_id=IDS[33], pressure=pressure)
        prefill = (events(1.0, allocated={pid(64): [1, 2], pid(96): [3, 4, 5]}) + GOOD_PREFILL
                   + events(3.0, finished=[pid(64), pid(96)]) + events(10.5, sends_done=[pid(64)])
                   + events(10.6, allocated={pid(33): [2, 1]}) + events(11.5, sends_done=[pid(96)]))
        prefill = prefill.replace(GOOD_PREFILL, GOOD_PREFILL + step_log((pid(33), 64, 0, 64, 10.6, 10.7)))
        decode = (REAL_DECODE + events(12.0, allocated={did(33): [1, 2]}, reads_posted=[did(33)])
                  + events(12.1, reads_done=[did(33)]) + step_log((did(33), 64, 63, 1, 12.2, 12.3))
                  + events(12.4, finished=[did(33)]))
        prefill += events(12.15, finished=[pid(33)], sends_done=[pid(33)])
        checks = verify([*held, blocked], prefill, decode, scenario="pressure", real=True)
        self.assertEqual(checks[2]["reused_held_blocks"], {IDS[64]: [1, 2]})
        self.assertAlmostEqual(checks[2]["allocated_after_release_s"], 0.1)
        with self.assertRaisesRegex(ValueError, "did not wait"):
            verify([*held, dict(blocked, pressure={"still_queued": False, "prefill_waiting": 0.0})],
                   prefill, decode, scenario="pressure", real=True)
        # Released before decode asked: the lease, not the READ, freed them.
        early = [dict(held[0], t_sent=10.55), held[1]]
        with self.assertRaisesRegex(ValueError, "before decode asked"):
            verify([*early, blocked], prefill, decode, scenario="pressure", real=True)

    def test_summary_counts_reuse_and_reads_native_metrics(self) -> None:
        from src.backends.vllm.parser import check_scenario

        checks = verify([record(64), record(96)], GOOD_PREFILL, CHECKED_DECODE, kv_check=True)
        summary = summarize(checks, METRICS, METRICS)
        self.assertEqual(summary["outcomes"], {"transferred": 2})
        self.assertEqual(summary["kv_bytes_checked"], 2 * 262144)
        # Block 2 served both requests.
        self.assertEqual((summary["decode_blocks_used"], summary["decode_blocks_reused"]), (3, 1))
        self.assertEqual(summary["decode_metrics"]["nixl_bytes"], 600000)
        good = {"addresses": {"target_address": "10.0.0.1"},
                "decode_connections": ["10.0.0.2:40000->10.0.0.1:41000"]}
        check_scenario("proxy", summary, good)
        with self.assertRaisesRegex(ValueError, "no connection to prefill's address"):
            check_scenario("proxy", summary, dict(good, decode_connections=["10.0.0.2:1->10.42.0.9:2"]))
        failing = summarize(checks, METRICS, METRICS.replace("transfers_total{engine=\"0\"} 0", "transfers_total{engine=\"0\"} 2"))
        with self.assertRaisesRegex(ValueError, "failed transfers"):
            check_scenario("proxy", failing, good)
        few = summarize(checks, METRICS, METRICS.replace("600000", "1000"))
        with self.assertRaisesRegex(ValueError, "fewer than"):
            check_scenario("proxy", few, good)
        busy = summarize(checks, METRICS, METRICS.replace("usage_perc{engine=\"0\"} 0.0", "usage_perc{engine=\"0\"} 0.25"))
        with self.assertRaisesRegex(ValueError, "still holds KV blocks"):
            check_scenario("cancel", busy, good)
        # Metrics that could not be read prove nothing about cleanup.
        for scenario in ("proxy", "cancel"):
            with self.subTest(scenario=scenario), self.assertRaisesRegex(ValueError, "no KV cache usage from decode"):
                check_scenario(scenario, summarize(checks, METRICS, ""), good)
        with self.assertRaisesRegex(ValueError, "no KV cache usage from prefill"):
            check_scenario("cancel", summarize(checks, "", METRICS), good)
        failed = summarize(checks, "", METRICS.replace("transfers_total{engine=\"0\"} 0", "transfers_total{engine=\"0\"} 2"))
        with self.assertRaisesRegex(ValueError, "not healthy"):
            check_scenario("producer-failure", failed, {"decode_healthy_after": False})
        with self.assertRaisesRegex(ValueError, "counted no failed transfer"):
            check_scenario("producer-failure", summary, {"decode_healthy_after": True})
        check_scenario("producer-failure", failed, {"decode_healthy_after": True})

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
        self.assertEqual(command["summary"]["outcomes"], {"transferred": 2})
        self.assertEqual(command["summary"]["decode_metrics"]["nixl_failed_transfers"], 0)
        self.assertEqual(command["decode_connections"], ["10.0.0.2:40000->10.0.0.1:41000"])
        observations = VllmParser().parse(run)
        self.assertEqual([(o.metric, o.dimensions["prompt_length"], o.value) for o in observations],
                         [("latency", 64, 0.2), ("latency", 96, 0.2)])

    def test_scenarios_reach_the_helper(self) -> None:
        kubectl = FakeKubectl()
        resources = BackendResources(names={
            "initiator": "vllm-pd-decode-t1", "target": "vllm-pd-prefill-t1",
            "initiator_address": "10.0.0.2", "target_address": "10.0.0.1",
        })
        self.runner(kubectl).execute(self.config(repeat=2), resources, self.directory / "raw" / "t1")
        (call,) = [c for c in kubectl.calls if "scenario" in c]
        helper = call[call.index("scenario"):]
        self.assertEqual(helper[1], "proxy")
        self.assertEqual(helper[helper.index("--prompt-lengths") + 1], "64,96,64,96")
        self.assertEqual(helper[helper.index("--decode") + 1], "http://10.0.0.2:8000")

    def test_execute_fails_when_decode_recomputes(self) -> None:
        kubectl = FakeKubectl(decode_steps=step_log((did(64), 64, 0, 64), (did(96), 96, 95, 1)))
        with self.assertRaisesRegex(RuntimeError, "was fallback"):
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


class PodHelperScenarioTests(unittest.TestCase):
    """The helper's two hops against local stand-ins for prefill and decode."""

    def setUp(self) -> None:
        import http.server
        import threading

        self.seen: list[tuple[str, str | None, dict]] = []
        seen = self.seen

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                role = "prefill" if body.get("kv_transfer_params", {}).get("do_remote_decode") else "decode"
                seen.append((role, self.headers.get("X-Request-Id"), body))
                if role == "decode" and body["prompt"][0] == 99:
                    import time
                    time.sleep(0.5)  # long enough for the client to give up
                request_id = self.headers.get("X-Request-Id")
                reply = {"id": f"cmpl-{request_id}", "choices": [{"token_ids": [70], "finish_reason": "length"}],
                         "usage": {"prompt_tokens": len(body["prompt"])}}
                if role == "prefill":
                    reply["kv_transfer_params"] = {"remote_block_ids": [[3, 4]], "remote_request_id": request_id}
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except OSError:
                    pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()

    def test_two_hops_hand_prefill_params_to_decode(self) -> None:
        from src.backends.vllm import pod_helper

        record = pod_helper.two_hops(self.url, self.url, "m", 0, 40, 1, timeout=5)
        (prefill_role, prefill_id, prefill_body), (decode_role, decode_id, decode_body) = self.seen
        self.assertEqual((prefill_role, decode_role), ("prefill", "decode"))
        # One request id on both hops, as vLLM's proxy does: it links the step logs.
        self.assertEqual(prefill_id, decode_id)
        self.assertEqual(record["request_id"], prefill_id)
        self.assertEqual(prefill_body["max_tokens"], 1)
        self.assertEqual(decode_body["kv_transfer_params"], {"remote_block_ids": [[3, 4]], "remote_request_id": prefill_id})
        self.assertLessEqual(record["prefill"]["t_received"], record["t_sent"])
        self.assertNotIn("error", record)

    def test_cancel_drops_the_decode_connection(self) -> None:
        from src.backends.vllm import pod_helper

        # Prompt token 99 (index 89) makes the stand-in decode slow.
        record = pod_helper.two_hops(self.url, self.url, "m", 89, 40, 1, timeout=5, decode_timeout=0.05)
        self.assertTrue(record["timed_out"])
        self.assertEqual(record["cancel_after_s"], 0.05)
        self.assertLess(record["elapsed_s"], 0.4)
