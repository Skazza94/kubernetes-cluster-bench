#!/usr/bin/env python3
"""Runs inside the vLLM pods: model fixture, readiness wait and the request client.

Shipped to the pods through the run's ConfigMap and executed with the image's
Python. ``prepare`` needs the image's tokenizers/transformers; the other
commands use only the standard library. Results are printed as JSON.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


VOCAB_SIZE = 128
FIRST_PROMPT_TOKEN = 10
FIRST_OUTPUT_TOKEN = 70


def prepare(directory: Path, max_model_len: int) -> None:
    """Write a tiny Llama config and word-level tokenizer: no weights, no downloads."""

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": VOCAB_SIZE,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": max_model_len,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    vocab.update({f"token{i}": i for i in range(4, VOCAB_SIZE)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=max_model_len,
        # The requests use completions, but the API renderer also warms up chat.
        chat_template="{% for m in messages %}{{ m['content'] }} {% endfor %}",
    )
    tokenizer.save_pretrained(str(directory))
    print(json.dumps({"prepared": str(directory)}))


def wait_for_url(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            with urlopen(url, timeout=min(5.0, remaining)) as response:
                response.read()
                if response.status == 200:
                    print(json.dumps({"ready": url}))
                    return
                last_error = f"HTTP {response.status}"
        except (URLError, TimeoutError, OSError) as error:
            last_error = str(error)
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise SystemExit(f"timed out after {timeout:g}s waiting for {url}: {last_error}")


def payload(model: str, index: int, prompt_length: int, output_tokens: int) -> dict:
    """A completion whose prompt and output token IDs are fixed (no sampling)."""

    return {
        "model": model,
        "prompt": [FIRST_PROMPT_TOKEN + index % 100] * prompt_length,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "stream": False,
        "add_special_tokens": False,
        "return_token_ids": True,
        "trace_decode_token_ids": [FIRST_OUTPUT_TOKEN + i for i in range(output_tokens)],
    }


def post(url: str, body: dict, timeout: float, request_id: str | None = None) -> dict:
    """POST JSON; returns the response, or an ``error`` entry. Times are this
    pod's monotonic clock (``t_sent``/``t_received``)."""

    headers = {"Content-Type": "application/json"}
    if request_id is not None:
        headers["X-Request-Id"] = request_id
    request = Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    result: dict = {"t_sent": time.monotonic()}
    try:
        with urlopen(request, timeout=timeout) as response:
            result["response"] = json.load(response)
    except HTTPError as error:
        result["error"] = f"HTTP {error.code}: {error.read().decode(errors='replace')[:500]}"
    except (URLError, TimeoutError, OSError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        # The client gave up (a cancellation when intended), as opposed to
        # an answer from the server.
        result["timed_out"] = isinstance(getattr(error, "reason", error), TimeoutError)
    result["t_received"] = time.monotonic()
    return result


def record_of(index: int, prompt_length: int, output_tokens: int, result: dict,
              request_id: str | None = None) -> dict:
    """One request's record. ``request_id`` is the uuid in both servers' request
    ids (the proxy's own, read back from the response id, when not given)."""

    response = result.get("response")
    if request_id is None and response is not None:
        request_id = str(response.get("id", "")).removeprefix("cmpl-")
    return {
        "index": index,
        "request_id": request_id,
        "prompt_length": prompt_length,
        "expected_token_ids": [FIRST_OUTPUT_TOKEN + i for i in range(output_tokens)],
        "t_sent": result["t_sent"],
        "t_received": result["t_received"],
        "elapsed_s": result["t_received"] - result["t_sent"],
        **{key: result[key] for key in ("response", "error", "timed_out") if key in result},
    }


def prefill_hop(prefill: str, model: str, index: int, prompt_length: int,
                timeout: float) -> dict:
    """The first hop of vLLM's toy proxy: a 1-token prefill with
    do_remote_decode. Prefill then holds the request's blocks for decode."""

    request_id = str(uuid.uuid4())
    body = dict(payload(model, index, prompt_length, 1), kv_transfer_params={
        "do_remote_decode": True, "do_remote_prefill": False, "remote_engine_id": None,
        "remote_block_ids": None, "remote_host": None, "remote_port": None,
    })
    result = post(f"{prefill}/v1/completions", body, timeout, request_id)
    hop = {"request_id": request_id, "index": index, "prompt_length": prompt_length,
           "t_sent": result["t_sent"], "t_received": result["t_received"],
           "kv_transfer_params": (result.get("response") or {}).get("kv_transfer_params")}
    if "error" in result:
        hop["error"] = result["error"]
    return hop


def decode_hop(decode: str, model: str, hop: dict, output_tokens: int, timeout: float,
               decode_timeout: float | None = None) -> dict:
    """The second hop: the decode request with prefill's kv_transfer_params.
    ``decode_timeout`` closes the decode connection early (a cancellation)."""

    body = payload(model, hop["index"], hop["prompt_length"], output_tokens)
    if hop["kv_transfer_params"]:
        body["kv_transfer_params"] = hop["kv_transfer_params"]
    result = post(f"{decode}/v1/completions", body, decode_timeout or timeout, hop["request_id"])
    record = record_of(hop["index"], hop["prompt_length"], output_tokens, result, hop["request_id"])
    record["prefill"] = {key: value for key, value in hop.items()
                         if key not in ("request_id", "index", "prompt_length")}
    if decode_timeout is not None:
        record["cancel_after_s"] = decode_timeout
    return record


def two_hops(prefill: str, decode: str, model: str, index: int, prompt_length: int,
             output_tokens: int, timeout: float, decode_timeout: float | None = None,
             before_decode=None) -> dict:
    """What vLLM's toy proxy does, step by step; ``before_decode`` runs
    between the two hops."""

    hop = prefill_hop(prefill, model, index, prompt_length, timeout)
    if before_decode is not None:
        before_decode()
    return decode_hop(decode, model, hop, output_tokens, timeout, decode_timeout)


def metric(base: str, name: str, timeout: float = 10) -> float | None:
    """A server's Prometheus metric, summed over its label sets."""

    try:
        with urlopen(f"{base}/metrics", timeout=timeout) as response:
            text = response.read().decode(errors="replace")
    except (URLError, TimeoutError, OSError):
        return None
    values = [float(line.rpartition(" ")[2]) for line in text.splitlines()
              if line.split("{", 1)[0].split(" ", 1)[0] == name]
    return sum(values) if values else None


def processes_table() -> dict[int, tuple[int, str]]:
    """This pod's processes: pid -> (parent pid, command line)."""

    table = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parent = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, ValueError, IndexError):
            continue
        table[int(entry.name)] = (parent, command)
    return table


def signal_processes(pids: list[int], number: int) -> list[int]:
    signalled = []
    for pid in pids:
        try:
            os.kill(pid, number)
            signalled.append(pid)
        except ProcessLookupError:
            pass
    return signalled


def other_processes() -> list[int]:
    """Every process in this pod but PID 1 (the pod's idle command) and this
    helper and its parents: the prefill server, its engine and the proxy."""

    table = processes_table()
    keep = {1}
    pid = os.getpid()
    while pid > 1 and pid not in keep:
        keep.add(pid)
        pid = table.get(pid, (1, ""))[0]
    return sorted(pid for pid in table if pid not in keep)


def server_processes() -> list[int]:
    """The vLLM server and every process under it (engine core, helpers)."""

    table = processes_table()
    found = {pid for pid, (_, command) in table.items()
             if " serve " in f" {command} " and "vllm" in command}
    while True:
        children = {pid for pid, (parent, _) in table.items() if parent in found} - found
        if not children:
            return sorted(found)
        found |= children


# cancel: the client gives up on a READ that cannot finish while the producer
# is stopped, and the producer resumes only after decode had time to abort.
CANCEL_DURING_READ_S = 1.0
# Requests of decode's own, sent this long into the pause: before and after
# the client gives up.
BYSTANDERS_AFTER_S = (0.3, CANCEL_DURING_READ_S + 0.3)
# pressure: how long the request that does not fit must stay queued, and at
# most how many requests hold blocks (a guard; the pool decides).
PRESSURE_WAIT_S = 2.0
MAX_HELD = 64


def cancel_during_read(prefill: str, decode: str, model: str, index: int, prompt_length: int,
                       output_tokens: int, timeout: float, pause_after_cancel: float) -> dict:
    """A decode request dropped while its READ is outstanding: prefill's
    server is stopped (SIGSTOP) after the handoff, so the READ decode posts
    cannot complete; the client disconnects, and prefill resumes (SIGCONT)
    ``pause_after_cancel`` seconds after the client gave up. vLLM keeps the
    aborted request's blocks until the READ ends, so decode's KV usage stays
    above 0 until then. Meanwhile bystanders, requests decode computes itself,
    show whether decode keeps serving while one producer is stuck."""

    paused: list[int] = []
    bystanders: list[dict] = [{} for _ in BYSTANDERS_AFTER_S]

    def send_bystander(slot: int) -> None:
        time.sleep(BYSTANDERS_AFTER_S[slot])
        number = index + 50 + slot
        request_id = str(uuid.uuid4())
        body = payload(model, number, prompt_length, output_tokens)
        result = post(f"{decode}/v1/completions", body, timeout, request_id)
        bystanders[slot].update(record_of(number, prompt_length, output_tokens, result, request_id))

    threads = [threading.Thread(target=send_bystander, args=(slot,), daemon=True)
               for slot in range(len(BYSTANDERS_AFTER_S))]

    def pause() -> None:
        paused.extend(signal_processes(server_processes(), signal.SIGSTOP))
        for thread in threads:
            thread.start()

    resumed = None
    try:
        record = two_hops(prefill, decode, model, index, prompt_length, output_tokens, timeout,
                          decode_timeout=CANCEL_DURING_READ_S, before_decode=pause)
        time.sleep(pause_after_cancel)
        usage = metric(decode, "vllm:kv_cache_usage_perc")
    finally:
        signal_processes(paused, signal.SIGCONT)
        resumed = time.monotonic()
    for thread in threads:
        thread.join(timeout)
    record["producer_paused"] = {
        "pids": paused,
        "decode_kv_usage_before_resume": usage,
        "t_resumed": resumed,
    }
    record["bystanders"] = bystanders
    return record


def hold_under_pressure(prefill: str, decode: str, model: str, prompt_length: int,
                        output_tokens: int, timeout: float) -> list[dict]:
    """Prefill's blocks stay reserved for decode until it reads them, even
    when another request needs them: prefill hops fill prefill's KV pool (no
    decode yet), one more prefill must queue, and it runs only once decode
    reads a held request and prefill frees, then reuses, those blocks."""

    warm_up = two_hops(prefill, decode, model, 0, prompt_length, output_tokens, timeout)
    held: list[dict] = []
    per_request = None
    while len(held) < MAX_HELD:
        time.sleep(0.1)  # let prefill publish the usage of the last hop
        usage = metric(prefill, "vllm:kv_cache_usage_perc")
        if usage is None:
            raise SystemExit("cannot read prefill's KV cache usage")
        if held and per_request is None:
            per_request = usage - before
            if per_request <= 0:
                raise SystemExit("prefill's KV cache usage did not grow with a held request")
        if per_request is not None and 1 - usage < per_request - 1e-9:
            break
        before = usage
        hop = prefill_hop(prefill, model, 1 + len(held), prompt_length, timeout)
        if hop.get("error") or not hop["kv_transfer_params"]:
            raise SystemExit(f"prefill failed while filling its KV pool: {hop}")
        held.append(hop)
    pressure: dict = {"held": len(held), "usage_held": usage}
    with ThreadPoolExecutor(max_workers=1) as pool:
        blocked = pool.submit(prefill_hop, prefill, model, 1 + len(held), prompt_length, timeout)
        time.sleep(PRESSURE_WAIT_S)
        pressure["queued_after_s"] = PRESSURE_WAIT_S
        pressure["still_queued"] = not blocked.done()
        pressure["prefill_waiting"] = metric(prefill, "vllm:num_requests_waiting")
        records = [dict(decode_hop(decode, model, hop, output_tokens, timeout), held=True)
                   for hop in held]
        last = blocked.result()
    records.append(dict(decode_hop(decode, model, last, output_tokens, timeout), pressure=pressure))
    return [warm_up, *records]


def run_scenario(scenario: str, proxy: str, prefill: str, decode: str, model: str,
                 prompt_lengths: list[int], output_tokens: int, concurrency: int,
                 timeout: float, cancel_delays: list[float], pause_after_cancel: float = 1.0) -> None:
    """Send the scenario's requests and print ``{"scenario", "requests", ...}``."""

    out: dict = {"scenario": scenario, "concurrency": concurrency}
    jobs = list(enumerate(prompt_lengths))
    if scenario == "proxy":
        def one(job):
            index, length = job
            body = payload(model, index, length, output_tokens)
            return record_of(index, length, output_tokens,
                             post(f"{proxy}/v1/completions", body, timeout))
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            out["requests"] = list(pool.map(one, jobs))
    elif scenario == "handoff":
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            out["requests"] = list(pool.map(
                lambda job: two_hops(prefill, decode, model, job[0], job[1], output_tokens, timeout),
                jobs))
    elif scenario == "cancel":
        # Decode connections closed at increasing delays (a request that
        # completes first is checked as a normal one), one closed while its
        # READ is outstanding, then normal requests.
        cancelled = [
            two_hops(prefill, decode, model, index, prompt_lengths[index % len(prompt_lengths)],
                     output_tokens, timeout, decode_timeout=delay)
            for index, delay in enumerate(cancel_delays)
        ]
        offset = len(cancel_delays)
        cancelled.append(cancel_during_read(prefill, decode, model, offset, prompt_lengths[0],
                                            output_tokens, timeout, pause_after_cancel))
        follow_up = [
            two_hops(prefill, decode, model, offset + 1 + index, length, output_tokens, timeout)
            for index, length in jobs
        ]
        out["requests"] = cancelled + follow_up
    elif scenario == "producer-failure":
        # A normal request first, so decode's handshake and hp_tcp lanes to the
        # producer exist; then one whose producer dies after handing its blocks over.
        warm_up = two_hops(prefill, decode, model, 0, prompt_lengths[0], output_tokens, timeout)
        killed: list[int] = []
        record = two_hops(prefill, decode, model, 1, prompt_lengths[0], output_tokens, timeout,
                          before_decode=lambda: killed.extend(
                              signal_processes(other_processes(), signal.SIGKILL)))
        record["producer_killed"] = killed
        out["requests"] = [warm_up, record]
        try:
            with urlopen(f"{decode}/health", timeout=10) as response:
                out["decode_healthy_after"] = response.status == 200
        except (URLError, TimeoutError, OSError) as error:
            out["decode_healthy_after"] = False
            out["decode_health_error"] = str(error)
    elif scenario == "pressure":
        out["requests"] = hold_under_pressure(
            prefill, decode, model, prompt_lengths[0], output_tokens, timeout)
    else:
        raise SystemExit(f"unknown scenario {scenario}")
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")


def fetch(url: str, timeout: float) -> None:
    """Print a URL's body (e.g. /metrics), or nothing if it cannot be read."""

    try:
        with urlopen(url, timeout=timeout) as response:
            sys.stdout.write(response.read().decode(errors="replace"))
    except (URLError, TimeoutError, OSError) as error:
        sys.stderr.write(f"cannot read {url}: {error}\n")


# TCP counters (/proc/net/snmp) the profiler records.
TCP_COUNTERS = ("ActiveOpens", "PassiveOpens", "InSegs", "OutSegs", "RetransSegs", "InErrs", "OutRsts")


def _read(path: str | Path) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def cgroup_usage_usec() -> int | None:
    """This container's CPU time (cgroup v2, else v1)."""

    for line in _read("/sys/fs/cgroup/cpu.stat").splitlines():
        if line.startswith("usage_usec "):
            return int(line.split()[1])
    for directory in ("/sys/fs/cgroup/cpu,cpuacct", "/sys/fs/cgroup/cpuacct"):
        usage = _read(f"{directory}/cpuacct.usage").strip()
        if usage:
            return int(usage) // 1000
    return None


def cgroup_throttling() -> dict[str, int] | None:
    """Cgroup v2 CFS counters, or unavailable if any counter cannot be read."""
    names = ("nr_periods", "nr_throttled", "throttled_usec")
    fields = dict(line.split() for line in _read("/sys/fs/cgroup/cpu.stat").splitlines())
    return {name:int(fields[name]) for name in names} if all(name in fields for name in names) else None


def host_cpu() -> list[int]:
    """The node's CPU ticks (/proc/stat is not per container): user, nice,
    system, idle, iowait, irq, softirq, steal."""

    fields = _read("/proc/stat").split("\n", 1)[0].split()
    return [int(value) for value in fields[1:9]]


def interfaces() -> dict[str, list[int]]:
    """This network namespace's interfaces but lo: rx/tx bytes and packets."""

    counters = {}
    for name in sorted(os.listdir("/sys/class/net")):
        if name == "lo":
            continue
        base = f"/sys/class/net/{name}/statistics"
        values = [_read(f"{base}/{counter}").strip()
                  for counter in ("rx_bytes", "rx_packets", "tx_bytes", "tx_packets")]
        if all(values):
            counters[name] = [int(value) for value in values]
    return counters


def tcp_counters() -> dict[str, int]:
    rows = [line.split()[1:] for line in _read("/proc/net/snmp").splitlines() if line.startswith("Tcp:")]
    if len(rows) < 2:
        return {}
    return {name: int(value) for name, value in zip(*rows[:2]) if name in TCP_COUNTERS}


def thread_counters() -> dict[int, list]:
    """Every thread in this pod: [pid, name, utime, stime (ticks), voluntary
    and involuntary context switches, the CPU it last ran on, the CPUs it may
    run on]."""

    threads = {}
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            tasks = list((process / "task").iterdir())
        except OSError:
            continue
        for task in tasks:
            stat = _read(task / "stat")
            if ")" not in stat:
                continue
            name = stat[stat.index("(") + 1:stat.rindex(")")]
            fields = stat[stat.rindex(")") + 2:].split()  # from field 3 (state)
            switches, allowed = [None, None], None
            for line in _read(task / "status").splitlines():
                if line.startswith("voluntary_ctxt_switches:"):
                    switches[0] = int(line.split()[1])
                elif line.startswith("nonvoluntary_ctxt_switches:"):
                    switches[1] = int(line.split()[1])
                elif line.startswith("Cpus_allowed_list:"):
                    allowed = line.split(":", 1)[1].strip()
            try:
                if None in switches or allowed is None:
                    continue
                threads[int(task.name)] = [int(process.name), name, int(fields[11]), int(fields[12]),
                                           *switches, int(fields[36]), allowed, int(fields[19])]
            except (IndexError, ValueError):
                continue
    return threads


def process_counters() -> list[list]:
    """Process CPU totals and identities; process totals already include threads."""
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat.rsplit(")", 1)[1].split()
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            rows.append([int(entry.name), int(fields[19]), int(fields[1]), command,
                         int(fields[11]), int(fields[12]), stat[stat.index("(")+1:stat.rindex(")")]])
        except (OSError, ValueError, IndexError):
            continue
    return rows


def profile(interval: float, duration: float) -> None:
    """Print this pod's CPU and network counters as JSON lines every
    ``interval`` seconds, stamped with the node's monotonic clock (the step
    logs' clock), for at most ``duration`` seconds. A first line gives the
    clock ticks per second and this profiler's own pid. Each sample has the
    container's CPU (``cg``, usec), its CFS throttling counters (``throttle``), the node's (``host``, ticks), each
    interface's and TCP's counters, and (``th``) the threads whose counters
    changed since the last sample, as [tid, *thread_counters(), wchan]: the
    kernel function the thread was waiting in when sampled ("0" if running),
    which tells threads apart when their names do not. Thread start ticks
    distinguish reused TIDs. `alive` lists the current thread inventory;
    `ps` records process identities and CPU totals independently of threads."""

    mtu = {name: int(_read(f"/sys/class/net/{name}/mtu") or 0) for name in interfaces()}
    print(json.dumps({"hz": os.sysconf("SC_CLK_TCK"), "self": os.getpid(), "interval": interval, "clock": "node_monotonic",
                      "boot_id": _read("/proc/sys/kernel/random/boot_id").strip(),
                      "mtu": mtu}), flush=True)
    last: dict[int, list] = {}
    start = time.monotonic()
    deadline = start
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    previous_handler = signal.signal(signal.SIGTERM, stop)
    try:
        while deadline - start <= duration:
            threads = thread_counters()
            changed = [[tid, *values[:8], _read(f"/proc/{values[0]}/task/{tid}/wchan").strip() or None, values[8]]
                       for tid, values in threads.items() if last.get(tid) != values]
            last = threads
            counters = {"cg": cgroup_usage_usec(), "throttle": cgroup_throttling(), "host": host_cpu(), "net": interfaces(),
                        "tcp": tcp_counters(), "th": changed, "alive": list(threads), "ps": process_counters()}
            print(json.dumps({"t": time.monotonic(), **counters},
                             separators=(",", ":")), flush=True)
            if stopping:
                break
            deadline += interval
            time.sleep(max(0.0, deadline - time.monotonic()))
    except BrokenPipeError:
        pass  # the runner stopped reading
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


PROFILER_COMMAND = "/bench/pod_helper.py profile"


def stop_profilers(timeout: float = 5.0) -> dict:
    """Stop every profiler in this pod (SIGTERM, then SIGKILL) and confirm
    none is left: one surviving a run would add its CPU to the next."""

    def running() -> list[int]:
        return [pid for pid, (_, command) in processes_table().items()
                if pid != os.getpid() and PROFILER_COMMAND in command]

    stopped = running()
    signal_processes(stopped, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while running() and time.monotonic() < deadline:
        time.sleep(0.05)
    signal_processes(running(), signal.SIGKILL)
    time.sleep(0.1)
    return {"stopped": stopped, "remaining": running()}


def affinity() -> dict:
    """Where this pod's processes may run: the container's CPU limit and
    cpuset, and each process's allowed CPUs."""

    processes = []
    for pid, (_, command) in sorted(processes_table().items()):
        allowed = next((line.split(":", 1)[1].strip()
                        for line in _read(f"/proc/{pid}/status").splitlines()
                        if line.startswith("Cpus_allowed_list:")), None)
        processes.append({"pid": pid, "command": command.strip()[:120], "cpus": allowed})
    return {
        "cpu_count": os.cpu_count(),
        "cpu.max": _read("/sys/fs/cgroup/cpu.max").strip() or None,
        "cpuset.cpus.effective": _read("/sys/fs/cgroup/cpuset.cpus.effective").strip() or None,
        "processes": processes,
    }


def tcp_connections() -> list[str]:
    """This network namespace's established IPv4 TCP connections, local->remote."""

    connections = []
    try:
        lines = Path("/proc/net/tcp").read_text().splitlines()[1:]
    except OSError:
        return connections
    for line in lines:
        fields = line.split()
        if len(fields) < 4 or fields[3] != "01":  # 01: ESTABLISHED
            continue
        ends = []
        for end in fields[1:3]:
            address, port = end.split(":")
            octets = [str(int(address[i:i + 2], 16)) for i in range(6, -1, -2)]
            ends.append(f"{'.'.join(octets)}:{int(port, 16)}")
        connections.append(f"{ends[0]}->{ends[1]}")
    return sorted(connections)



BENCH_PERCENTILES = "50,90,99"
BENCH_LOG_LINES = 40


def bench_command(base_url: str, model: str, tokenizer: str, concurrency: int, num_prompts: int,
                  dataset_name: str, dataset_path: str | None, request_rate: str, burstiness: float,
                  seed: int, bench_args: dict, trust_remote_code: bool, result_dir: str) -> list[str]:
    """vllm bench serve's command line: streaming completions, outputs as
    long as the dataset asks (ignore_eos), every request's result saved."""

    command = [
        str(Path(sys.executable).with_name("vllm")), "bench", "serve",
        "--backend", "openai", "--endpoint", "/v1/completions", "--base-url", base_url,
        "--model", model, "--tokenizer", tokenizer,
        "--dataset-name", dataset_name, "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--request-rate", request_rate, "--burstiness", str(burstiness), "--seed", str(seed),
        "--ignore-eos", "--disable-tqdm",
        "--ready-check-timeout-sec", "0", "--num-warmups", "0",
        "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", BENCH_PERCENTILES,
        "--save-result", "--save-detailed", "--result-dir", result_dir,
        "--result-filename", "bench.json",
    ]
    if dataset_path:
        command += ["--dataset-path", dataset_path]
    if trust_remote_code:
        command.append("--trust-remote-code")
    for name, value in bench_args.items():
        if value is True:
            command.append(f"--{name}")
        elif value is not False and value is not None:
            command += [f"--{name}", str(value)]
    return command


def bench_records(result: dict) -> list[dict]:
    """One record per request from vllm bench serve's detailed result, in
    seconds: time to first token, time per output token after it, and
    end-to-end latency."""

    names = ("input_lens", "output_lens", "ttfts", "latencies", "errors")
    if len({len(result[name]) for name in names}) != 1:
        raise ValueError("client detailed arrays have inconsistent lengths")
    records = []
    for index, (prompt, output, ttft, latency, error) in enumerate(zip(
            result["input_lens"], result["output_lens"], result["ttfts"],
            result["latencies"], result["errors"])):
        record: dict = {"index": index, "prompt_length": prompt, "output_tokens": output}
        if error:
            record["error"] = error
        else:
            record.update(ttft_s=ttft, latency_s=latency,
                          tpot_s=(latency - ttft) / (output - 1) if output > 1 else None)
        records.append(record)
    return records


def bench_window(result: dict) -> list[float] | None:
    """Successful workload bounds on this host's clock; unknown for failures.

    Failed requests may lack their end timestamp. Never shorten a mixed run
    to its successful subset and call that the workload's window.
    """
    if any(result.get("errors", [])):
        return None
    if time.get_clock_info("perf_counter").implementation != time.get_clock_info("monotonic").implementation:
        return None
    starts, latencies = result.get("start_times", []), result.get("latencies", [])
    if not starts or len(starts) != len(latencies):
        return None
    if any(not math.isfinite(t) or t <= 0 for t in starts) or any(not math.isfinite(t) or t < 0 for t in latencies):
        return None
    return [min(starts), max(t + latency for t, latency in zip(starts, latencies))]


def bench(artifact_directory: str = "/tmp/bench-workloads", **options) -> None:
    """Keep the complete client result and log; collection does not certify KV."""
    result_dir = Path(artifact_directory) / uuid.uuid4().hex
    result_dir.mkdir(parents=True, exist_ok=False)
    command = bench_command(**options, result_dir=str(result_dir))
    # Keep image CLI/datasets intact; instrument only its OpenAI request hook.
    command = [sys.executable, "/bench/ledger.py", "client", str(result_dir / "client-ledger.jsonl"), *command[1:]]
    started = time.monotonic()
    log = result_dir / "bench.log"
    with log.open("w") as out:
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            birth = int(Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        except (OSError, ValueError, IndexError):
            birth = None
        (Path(artifact_directory) / "client-process.json").write_text(json.dumps({
            "pid": process.pid, "start_ticks": birth, "process_group": process.pid, "command": command}))
        code = process.wait()
        finished = time.monotonic()
    result_path = result_dir / "bench.json"
    error = None
    try:
        result = json.loads(result_path.read_text()) if result_path.exists() else None
    except (OSError, ValueError) as failure:
        result, error = None, str(failure)
    payload = {"status": "collected_unvalidated", "exit_code": code, "command": command,
               "artifact_directory": str(result_dir), "bench": result,
               "result_error": error,
               "client": {"pids": [os.getpid(), process.pid], "started": started, "finished": finished,
                          "window": bench_window(result) if result else None,
                          "window_clock": "node_monotonic",
                          "execution_window": [started, finished]}}
    try:
        payload["requests"] = bench_records(result) if result else []
    except (ValueError, KeyError, TypeError) as failure:
        payload["requests"], payload["result_error"] = [], str(failure)
    # Persist before stdout or validation; a failed exec still leaves a path
    # known by the runner (it supplies a fresh artifact_directory).
    (Path(artifact_directory) / "last-client.json").write_text(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)
    if code:
        raise SystemExit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("prepare", help="create the local model config and tokenizer")
    fixture.add_argument("--directory", type=Path, required=True)
    fixture.add_argument("--max-model-len", type=int, required=True)
    ready = commands.add_parser("wait", help="wait for an HTTP endpoint to answer 200")
    ready.add_argument("--url", required=True)
    ready.add_argument("--timeout", type=float, required=True)
    client = commands.add_parser("scenario", help="send a scenario's requests and print the records")
    client.add_argument("name", choices=("proxy", "handoff", "cancel", "producer-failure",
                                         "pressure"))
    client.add_argument("--proxy", required=True, help="proxy base URL")
    client.add_argument("--prefill", required=True, help="prefill server base URL")
    client.add_argument("--decode", required=True, help="decode server base URL")
    client.add_argument("--model", required=True)
    client.add_argument("--prompt-lengths", required=True, help="comma-separated token counts")
    client.add_argument("--output-tokens", type=int, required=True)
    client.add_argument("--concurrency", type=int, required=True)
    client.add_argument("--timeout", type=float, required=True)
    client.add_argument("--cancel-delays", default="", help="comma-separated seconds (cancel)")
    client.add_argument("--pause-after-cancel", type=float, default=1.0,
                        help="seconds prefill stays stopped after the client gives up (cancel)")
    workload = commands.add_parser("bench", help="run vllm bench serve and print its records")
    workload.add_argument("--base-url", required=True, help="the proxy's base URL")
    workload.add_argument("--model", required=True)
    workload.add_argument("--tokenizer", required=True)
    workload.add_argument("--concurrency", type=int, required=True)
    workload.add_argument("--num-prompts", type=int, required=True)
    workload.add_argument("--dataset-name", required=True)
    workload.add_argument("--dataset-path")
    workload.add_argument("--request-rate", default="inf")
    workload.add_argument("--burstiness", type=float, default=1.0)
    workload.add_argument("--seed", type=int, default=0)
    workload.add_argument("--bench-args", default="{}", help="other options, as a JSON object")
    workload.add_argument("--trust-remote-code", action="store_true")
    workload.add_argument("--artifact-directory", default="/tmp/bench-workloads")
    reader = commands.add_parser("fetch", help="print a URL's body")
    reader.add_argument("--url", required=True)
    reader.add_argument("--timeout", type=float, default=10)
    commands.add_parser("connections", help="print the established TCP connections")
    commands.add_parser("clock", help="print this server's monotonic timestamp")
    profiler = commands.add_parser("profile", help="print CPU and network counters as JSON lines")
    profiler.add_argument("--interval", type=float, required=True)
    profiler.add_argument("--duration", type=float, required=True)
    commands.add_parser("affinity", help="print the CPU limits and affinities")
    commands.add_parser("stop-profilers", help="stop this pod's profilers and confirm it")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory, args.max_model_len)
    elif args.command == "wait":
        wait_for_url(args.url, args.timeout)
    elif args.command == "bench":
        bench(base_url=args.base_url, model=args.model, tokenizer=args.tokenizer,
              concurrency=args.concurrency, num_prompts=args.num_prompts,
              dataset_name=args.dataset_name, dataset_path=args.dataset_path,
              request_rate=args.request_rate, burstiness=args.burstiness, seed=args.seed,
              bench_args=json.loads(args.bench_args), trust_remote_code=args.trust_remote_code,
              artifact_directory=args.artifact_directory)
    elif args.command == "fetch":
        fetch(args.url, args.timeout)
    elif args.command == "connections":
        json.dump(tcp_connections(), sys.stdout)
        sys.stdout.write("\n")
    elif args.command == "clock":
        print(json.dumps({"t": time.monotonic()}))
    elif args.command == "profile":
        profile(args.interval, args.duration)
    elif args.command == "affinity":
        json.dump(affinity(), sys.stdout)
        sys.stdout.write("\n")
    elif args.command == "stop-profilers":
        result = stop_profilers()
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        if result["remaining"]:
            raise SystemExit(1)
    else:
        lengths = [int(value) for value in args.prompt_lengths.split(",")]
        delays = [float(value) for value in args.cancel_delays.split(",") if value]
        run_scenario(args.name, args.proxy, args.prefill, args.decode, args.model, lengths,
                     args.output_tokens, args.concurrency, args.timeout, delays,
                     args.pause_after_cancel)


if __name__ == "__main__":
    main()
