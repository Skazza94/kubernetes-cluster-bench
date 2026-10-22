#!/usr/bin/env python3
"""Runs inside the vLLM pods: model fixture, readiness wait and the request client.

Shipped to the pods through the run's ConfigMap and executed with the image's
Python. ``prepare`` needs the image's tokenizers/transformers; the other
commands use only the standard library. Results are printed as JSON.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import sys
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
        **{key: result[key] for key in ("response", "error") if key in result},
    }


def two_hops(prefill: str, decode: str, model: str, index: int, prompt_length: int,
             output_tokens: int, timeout: float, decode_timeout: float | None = None,
             before_decode=None) -> dict:
    """What vLLM's toy proxy does, step by step: a 1-token prefill with
    do_remote_decode, then the decode request with prefill's kv_transfer_params.
    ``decode_timeout`` closes the decode connection early (a cancellation);
    ``before_decode`` runs between the two hops."""

    request_id = str(uuid.uuid4())
    body = payload(model, index, prompt_length, output_tokens)
    first = dict(body, max_tokens=1, kv_transfer_params={
        "do_remote_decode": True, "do_remote_prefill": False, "remote_engine_id": None,
        "remote_block_ids": None, "remote_host": None, "remote_port": None,
    })
    prefill_result = post(f"{prefill}/v1/completions", first, timeout, request_id)
    hop = {"t_sent": prefill_result["t_sent"], "t_received": prefill_result["t_received"]}
    if "error" in prefill_result:
        hop["error"] = prefill_result["error"]
    params = (prefill_result.get("response") or {}).get("kv_transfer_params")
    hop["kv_transfer_params"] = params
    if before_decode is not None:
        before_decode()
    if params:
        body["kv_transfer_params"] = params
    decode_result = post(f"{decode}/v1/completions", body, decode_timeout or timeout, request_id)
    record = record_of(index, prompt_length, output_tokens, decode_result, request_id)
    record["prefill"] = hop
    if decode_timeout is not None:
        record["cancel_after_s"] = decode_timeout
    return record


def kill_other_processes() -> list[int]:
    """Kill every process in this pod but PID 1 (the pod's idle command) and
    this helper and its parents: the prefill server, its engine and the proxy."""

    keep = {1}
    pid = os.getpid()
    while pid > 1:
        keep.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    killed = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and int(entry.name) not in keep:
            try:
                os.kill(int(entry.name), signal.SIGKILL)
                killed.append(int(entry.name))
            except ProcessLookupError:
                pass
    return killed


def run_scenario(scenario: str, proxy: str, prefill: str, decode: str, model: str,
                 prompt_lengths: list[int], output_tokens: int, concurrency: int,
                 timeout: float, cancel_delays: list[float]) -> None:
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
        # Decode connections closed at increasing delays, then normal requests.
        cancelled = [
            two_hops(prefill, decode, model, index, prompt_lengths[index % len(prompt_lengths)],
                     output_tokens, timeout, decode_timeout=delay)
            for index, delay in enumerate(cancel_delays)
        ]
        for record in cancelled:
            record["cancelled"] = True
        offset = len(cancel_delays)
        follow_up = [
            two_hops(prefill, decode, model, offset + index, length, output_tokens, timeout)
            for index, length in jobs
        ]
        out["requests"] = cancelled + follow_up
    elif scenario == "producer-failure":
        # A request whose producer dies after handing its blocks over.
        killed: list[int] = []
        record = two_hops(prefill, decode, model, 0, prompt_lengths[0], output_tokens, timeout,
                          before_decode=lambda: killed.extend(kill_other_processes()))
        record["producer_killed"] = killed
        out["requests"] = [record]
        try:
            with urlopen(f"{decode}/health", timeout=10) as response:
                out["decode_healthy_after"] = response.status == 200
        except (URLError, TimeoutError, OSError) as error:
            out["decode_healthy_after"] = False
            out["decode_health_error"] = str(error)
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


CHECK_NOTIFICATION = b"transport-check-done"


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


def _pattern(size: int):
    import numpy as np

    return ((np.arange(size, dtype=np.uint64) * 31 + 7) % 251).astype(np.uint8)


def _send_json(connection: socket.socket, value: dict) -> None:
    connection.sendall(json.dumps(value).encode() + b"\n")


def _receive_json(connection: socket.socket) -> dict:
    data = b""
    while not data.endswith(b"\n"):
        chunk = connection.recv(65536)
        if not chunk:
            raise ConnectionError("peer closed the connection")
        data += chunk
    return json.loads(data)


def transport_check(role: str, backend: str, address: str, port: int, blocks: int,
                    block_bytes: int, timeout: float) -> None:
    """One small READ through a NIXL backend, as vLLM's pull connector does it.

    Both sides register a DRAM buffer with the backend and describe it as a
    strided (Nx5) list of ``blocks`` blocks. The target fills its buffer with a
    pattern and sends its agent metadata and buffer address over a TCP socket
    at ``address:port``; the initiator READs every block with a completion
    notification and checks every byte. Prints a JSON record either way.
    """

    import importlib.metadata

    import numpy as np
    from nixl._api import nixl_agent, nixl_agent_config

    size = blocks * block_bytes
    record: dict = {
        "role": role, "backend": backend, "blocks": blocks, "block_bytes": block_bytes,
        "nixl_version": importlib.metadata.version("nixl"),
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("MC_", "NIXL_")) or k in ("LD_LIBRARY_PATH",)},
    }
    agent = nixl_agent(f"transport-check-{role}", nixl_agent_config(backends=[backend]))
    record["plugins"] = agent.get_plugin_list()
    record["backend_params"] = agent.get_backend_params(backend)
    buffer = _pattern(size) if role == "target" else np.zeros(size, dtype=np.uint8)
    address_of_buffer = buffer.ctypes.data
    registration = agent.get_reg_descs([(address_of_buffer, size, 0, "")], "DRAM")
    agent.register_memory(registration, backends=[backend])
    strided = np.array([[address_of_buffer, block_bytes, 0, block_bytes, blocks]], dtype=np.uint64)
    deadline = time.monotonic() + timeout
    if role == "target":
        with socket.create_server((address, port)) as server:
            server.settimeout(timeout)
            connection, peer = server.accept()
            with connection:
                connection.settimeout(timeout)
                _send_json(connection, {
                    "metadata": base64.b64encode(agent.get_agent_metadata()).decode(),
                    "address": address_of_buffer,
                })
                notified = False
                while not notified and time.monotonic() < deadline:
                    for messages in agent.get_new_notifs().values():
                        notified = notified or CHECK_NOTIFICATION in messages
                    time.sleep(0.01)
                record["notified"] = notified
                record["initiator"] = _receive_json(connection)
                record["peer"] = peer[0]
                record["tcp_connections"] = tcp_connections()
        record["ok"] = bool(notified and record["initiator"].get("ok"))
    else:
        while True:
            try:
                connection = socket.create_connection((address, port), timeout=5)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        with connection:
            connection.settimeout(timeout)
            offer = _receive_json(connection)
            remote = agent.add_remote_agent(base64.b64decode(offer["metadata"]))
            local_side = agent.prep_xfer_dlist("NIXL_INIT_AGENT", strided, "DRAM")
            remote_side = agent.prep_xfer_dlist(
                remote,
                np.array([[offer["address"], block_bytes, 0, block_bytes, blocks]], dtype=np.uint64),
                "DRAM",
            )
            indices = np.arange(blocks, dtype=np.int32)
            handle = agent.make_prepped_xfer("READ", local_side, indices, remote_side, indices,
                                             notif_msg=CHECK_NOTIFICATION)
            started = time.monotonic()
            state = agent.transfer(handle)
            while state not in ("DONE", "ERR") and time.monotonic() < deadline:
                state = agent.check_xfer_state(handle)
            record["state"] = state
            record["elapsed_s"] = time.monotonic() - started
            mismatched = int(np.count_nonzero(buffer != _pattern(size)))
            record["mismatched_bytes"] = mismatched
            record["ok"] = state == "DONE" and mismatched == 0
            # Which addresses the backend's connections use (they stay open).
            record["tcp_connections"] = tcp_connections()
            agent.release_xfer_handle(handle)
            _send_json(connection, {k: record[k] for k in ("state", "elapsed_s", "mismatched_bytes", "ok")})
    json.dump(record, sys.stdout)
    sys.stdout.write("\n")
    if not record["ok"]:
        raise SystemExit(1)


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
    client.add_argument("name", choices=("proxy", "handoff", "cancel", "producer-failure"))
    client.add_argument("--proxy", required=True, help="proxy base URL")
    client.add_argument("--prefill", required=True, help="prefill server base URL")
    client.add_argument("--decode", required=True, help="decode server base URL")
    client.add_argument("--model", required=True)
    client.add_argument("--prompt-lengths", required=True, help="comma-separated token counts")
    client.add_argument("--output-tokens", type=int, required=True)
    client.add_argument("--concurrency", type=int, required=True)
    client.add_argument("--timeout", type=float, required=True)
    client.add_argument("--cancel-delays", default="", help="comma-separated seconds (cancel)")
    reader = commands.add_parser("fetch", help="print a URL's body")
    reader.add_argument("--url", required=True)
    reader.add_argument("--timeout", type=float, default=10)
    commands.add_parser("connections", help="print the established TCP connections")
    check = commands.add_parser("transport-check", help="one small NIXL READ between two pods")
    check.add_argument("--role", choices=("target", "initiator"), required=True)
    check.add_argument("--backend", required=True)
    check.add_argument("--address", required=True, help="target's address (to listen on or reach)")
    check.add_argument("--port", type=int, required=True)
    check.add_argument("--blocks", type=int, default=64)
    check.add_argument("--block-bytes", type=int, default=64 * 1024)
    check.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if args.command == "transport-check":
        transport_check(args.role, args.backend, args.address, args.port, args.blocks,
                        args.block_bytes, args.timeout)
    elif args.command == "prepare":
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
                     args.output_tokens, args.concurrency, args.timeout, delays)


if __name__ == "__main__":
    main()
