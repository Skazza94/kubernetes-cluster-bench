"""Pod application instrumentation; native connector/transport code is untouched.

The native proxies choose a new UUID. Wrap their application-local UUID factory
and ASGI boundary to retain its relationship to the incoming client header.
The client wraps the installed benchmark request function, preserving its API,
dataset sampling, streaming, semaphore and timing behavior.
"""
from __future__ import annotations

import contextvars
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid


def append(path: str, record: dict) -> None:
    record = {**record, "t": time.monotonic(), "pid": os.getpid()}
    with open(path, "a", encoding="utf-8") as target:
        target.write(json.dumps(record, separators=(",", ":")) + "\n")


class ProxyLedger:
    def __init__(self, app, path: str):
        self.app, self.path = app, path
        self.client = contextvars.ContextVar("benchmark_client", default=None)

    def uuid4(self):
        value = uuid.uuid4()
        append(self.path, {"event": "mapped", "client_id": self.client.get(), "proxy_id": str(value)})
        return value

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in ("/v1/completions", "/v1/chat/completions"):
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        client = headers.get(b"x-request-id", b"").decode()
        token = self.client.set(client)
        status = None
        async def tracked(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
        try:
            return await self.app(scope, receive, tracked)
        finally:
            append(self.path, {"event": "finished", "client_id": client, "status": status})
            self.client.reset(token)


def client_wrapper(original, path: str):
    async def request(request_func_input, session, pbar=None):
        client_id = "kcb-" + uuid.uuid4().hex
        request_func_input.request_id = client_id
        append(path, {"event": "started", "client_id": client_id,
                      "prompt_length": request_func_input.prompt_len,
                      "requested_output": request_func_input.output_len,
                      "prompt": request_func_input.prompt,
                      "extra_body": request_func_input.extra_body,
                      "multi_modal_content": request_func_input.multi_modal_content,
                      "chat_messages": request_func_input.chat_messages})
        try:
            output = await original(request_func_input=request_func_input, session=session, pbar=pbar)
        except BaseException as error:
            append(path, {"event": "exception", "client_id": client_id, "error": str(error)})
            raise
        append(path, {"event": "finished", "client_id": client_id,
                      "start_time": output.start_time, "latency": output.latency,
                      "success": output.success, "error": output.error,
                      "prompt_length": output.prompt_len, "output_tokens": output.output_tokens})
        return output
    return request


def join_client(result: dict, ledger: list[dict]) -> list[dict]:
    """Join exact output.start_time values on the *same* client clock.

    No server-clock alignment or prompt/order heuristic is used. Collisions,
    extra requests (warmup/probes), failures and missing records fail closed.
    """
    finished = [r for r in ledger if r.get("event") == "finished"]
    started = [r["client_id"] for r in ledger if r.get("event") == "started"]
    times = result.get("start_times", [])
    if not times or len(set(times)) != len(times) or len(finished) != len(times):
        raise ValueError("ambiguous/incomplete client ledger or extra warmup/probe requests")
    index = {r["start_time"]: r for r in finished}
    if len(index) != len(finished) or set(index) != set(times):
        raise ValueError("client ledger does not match detailed result start times")
    ids = [r["client_id"] for r in finished]
    if len(set(ids)) != len(ids) or sorted(ids) != sorted(started):
        raise ValueError("missing or duplicated client request identity")
    records = []
    names = ("input_lens", "output_lens", "latencies", "errors", "ttfts")
    if any(len(result.get(name, [])) != len(times) for name in names):
        raise ValueError("incomplete detailed client arrays")
    for i, t in enumerate(times):
        row = index[t]
        if (not math.isfinite(t) or t <= 0
                or any(not math.isfinite(result[name][i]) or result[name][i] < 0 for name in ("latencies", "ttfts"))
                or result["ttfts"][i] > result["latencies"][i]):
            raise ValueError("invalid client timing evidence")
        if (row["latency"], row["prompt_length"], row["output_tokens"], row["error"]) != (
                result["latencies"][i], result["input_lens"][i], result["output_lens"][i], result["errors"][i]):
            raise ValueError("client ledger disagrees with detailed result")
        records.append({**row, "index": i, "ttft_s": result["ttfts"][i]})
    return records


def main():
    mode, path, *arguments = sys.argv[1:]
    if mode == "client":
        from vllm.benchmarks.lib.endpoint_request_func import ASYNC_REQUEST_FUNCS
        ASYNC_REQUEST_FUNCS["openai"] = client_wrapper(ASYNC_REQUEST_FUNCS["openai"], path)
        sys.argv = ["vllm", *arguments]
        from vllm.entrypoints.cli.main import main as cli
        cli()
    elif mode == "proxy":
        source, *arguments = arguments
        spec = importlib.util.spec_from_file_location("benchmark_native_proxy", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.argv = [source, *arguments]
        module.global_args = module.parse_args()
        ledger = ProxyLedger(module.app, path)
        module.uuid = SimpleNamespace(uuid4=ledger.uuid4)
        import uvicorn
        uvicorn.run(ledger, host=module.global_args.host, port=module.global_args.port)
    else:
        raise ValueError(f"unknown ledger mode {mode}")


if __name__ == "__main__":
    main()
