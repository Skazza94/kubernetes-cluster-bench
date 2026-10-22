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
from pathlib import Path
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
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory, args.max_model_len)
    elif args.command == "wait":
        wait_for_url(args.url, args.timeout)
    else:
        lengths = [int(value) for value in args.prompt_lengths.split(",")]
        requests(args.url, args.model, lengths, args.output_tokens, args.concurrency, args.timeout)


if __name__ == "__main__":
    main()
