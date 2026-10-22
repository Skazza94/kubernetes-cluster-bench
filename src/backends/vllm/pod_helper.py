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
import socket
import sys
import time
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


def send(url: str, model: str, index: int, prompt_length: int, output_tokens: int,
         timeout: float, origin: float) -> dict:
    """One completion request; prompt and output token IDs identify it in the step logs."""

    output_ids = [FIRST_OUTPUT_TOKEN + i for i in range(output_tokens)]
    payload = {
        "model": model,
        "prompt": [FIRST_PROMPT_TOKEN + index] * prompt_length,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "stream": False,
        "add_special_tokens": False,
        "return_token_ids": True,
        "trace_decode_token_ids": output_ids,
    }
    request = Request(url, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    record = {"index": index, "prompt_length": prompt_length,
              "expected_token_ids": output_ids, "started_s": started - origin}
    try:
        with urlopen(request, timeout=timeout) as response:
            record["response"] = json.load(response)
    except HTTPError as error:
        record["error"] = f"HTTP {error.code}: {error.read().decode(errors='replace')}"
    except (URLError, TimeoutError, OSError) as error:
        record["error"] = str(error)
    finished = time.monotonic()
    record["finished_s"] = finished - origin
    record["elapsed_s"] = finished - started
    return record


def requests(url: str, model: str, prompt_lengths: list[int], output_tokens: int,
             concurrency: int, timeout: float) -> None:
    """Send every prompt once, ``concurrency`` at a time, and print the records.

    Times are this pod's monotonic clock, relative to the batch start: client
    latency only, never compared with the other server's clock.
    """

    origin = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(send, url, model, index, length, output_tokens, timeout, origin)
            for index, length in enumerate(prompt_lengths)
        ]
        records = [future.result() for future in futures]
    json.dump({"concurrency": concurrency, "requests": records}, sys.stdout)
    sys.stdout.write("\n")


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
    client = commands.add_parser("requests", help="send the prompts and print the records")
    client.add_argument("--url", required=True)
    client.add_argument("--model", required=True)
    client.add_argument("--prompt-lengths", required=True, help="comma-separated token counts")
    client.add_argument("--output-tokens", type=int, required=True)
    client.add_argument("--concurrency", type=int, required=True)
    client.add_argument("--timeout", type=float, required=True)
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
    else:
        lengths = [int(value) for value in args.prompt_lengths.split(",")]
        requests(args.url, args.model, lengths, args.output_tokens, args.concurrency, args.timeout)


if __name__ == "__main__":
    main()
