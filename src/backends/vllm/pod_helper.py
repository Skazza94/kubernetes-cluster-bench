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
import os
from pathlib import Path
import signal
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


def processes() -> dict[int, tuple[int, str]]:
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

    table = processes()
    keep = {1}
    pid = os.getpid()
    while pid > 1 and pid not in keep:
        keep.add(pid)
        pid = table.get(pid, (1, ""))[0]
    return sorted(pid for pid in table if pid not in keep)


def server_processes() -> list[int]:
    """The vLLM server and every process under it (engine core, helpers)."""

    table = processes()
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
    reader = commands.add_parser("fetch", help="print a URL's body")
    reader.add_argument("--url", required=True)
    reader.add_argument("--timeout", type=float, default=10)
    commands.add_parser("connections", help="print the established TCP connections")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory, args.max_model_len)
    elif args.command == "wait":
        wait_for_url(args.url, args.timeout)
    elif args.command == "fetch":
        fetch(args.url, args.timeout)
    elif args.command == "connections":
        json.dump(tcp_connections(), sys.stdout)
        sys.stdout.write("\n")
    else:
        lengths = [int(value) for value in args.prompt_lengths.split(",")]
        delays = [float(value) for value in args.cancel_delays.split(",") if value]
        run_scenario(args.name, args.proxy, args.prefill, args.decode, args.model, lengths,
                     args.output_tokens, args.concurrency, args.timeout, delays,
                     args.pause_after_cancel)


if __name__ == "__main__":
    main()
