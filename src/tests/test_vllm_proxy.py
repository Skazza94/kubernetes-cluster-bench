"""Native proxy initialization with mocked upstream HTTP; no web server runs."""
from __future__ import annotations

import ast
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.backends.vllm import config, ledger, lifecycle
from src.backends.vllm.runner import VllmRunner
from src.benchmark.contracts import BenchmarkConfig


def mooncake_proxy(http):
    # Execute the native startup functions without importing optional web
    # packages. Route declarations are checked from this same source below.
    tree = ast.parse((config.FORK / config.MOONCAKE_PROXY).read_text())
    if http is not None:
        http.Limits = Mock()
    names = {"maybe_wrap_ipv6_address", "make_http_path", "prefiller_cycle",
             "get_prefiller_info", "lifespan"}
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    import ipaddress
    import itertools
    import urllib.parse
    scope = {"asyncio": asyncio, "asynccontextmanager": asynccontextmanager,
             "ipaddress": ipaddress, "itertools": itertools, "urllib": urllib,
             "httpx": http, "FastAPI": object, "Any": Any,
             "global_args": SimpleNamespace(prefill=[("http://prefill:8000", 8998)],
                                            decode=["http://decode:8000"])}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(config.MOONCAKE_PROXY), "exec"), scope)
    app = SimpleNamespace(state=SimpleNamespace())
    return scope, app, tree


async def readiness(wrapper):
    messages = []
    async def send(message):
        messages.append(message)
    async def receive():
        raise AssertionError("readiness must not read a generation payload")
    await wrapper({"type": "http", "method": "GET", "path": "/bench/ready"}, receive, send)
    return messages[0]["status"]


class ProxyReadinessTests(unittest.TestCase):
    def test_native_health_and_bootstrap_must_finish_before_readiness(self):
        async def run():
            health, bootstrap = asyncio.Event(), asyncio.Event()
            queried = asyncio.Event()
            async def get(url):
                if url == "/health":
                    await health.wait()
                else:
                    self.assertEqual(url, "http://prefill:8998/query")
                    queried.set()
                    await bootstrap.wait()
                return SimpleNamespace(raise_for_status=Mock(), json=lambda: {"0": {"engine_id": "engine-0"}})
            prefill = SimpleNamespace(get=AsyncMock(side_effect=get), aclose=AsyncMock())
            decode = SimpleNamespace(aclose=AsyncMock())
            native, app, _ = mooncake_proxy(SimpleNamespace(AsyncClient=Mock(side_effect=[prefill, decode])))
            wrapper = ledger.ProxyLedger(app, "unused-ledger", "mooncake")
            async with native["lifespan"](app):
                await asyncio.sleep(0)
                self.assertEqual(await readiness(wrapper), 503)
                health.set()
                await asyncio.wait_for(queried.wait(), 1)
                self.assertEqual(await readiness(wrapper), 503)
                bootstrap.set()
                await asyncio.wait_for(app.state.ready.wait(), 1)
                self.assertEqual(await readiness(wrapper), 200)
                self.assertEqual(app.state.prefill_clients[0]["dp_engine_id"], {0: "engine-0"})
                self.assertEqual(prefill.get.await_count, 2)
            prefill.aclose.assert_awaited_once()
            decode.aclose.assert_awaited_once()
        asyncio.run(run())

    def test_empty_or_incomplete_native_bootstrap_is_not_ready(self):
        async def run(data):
            response = SimpleNamespace(raise_for_status=Mock(), json=lambda: data)
            prefill = SimpleNamespace(get=AsyncMock(return_value=response), aclose=AsyncMock())
            decode = SimpleNamespace(aclose=AsyncMock())
            native, app, _ = mooncake_proxy(SimpleNamespace(AsyncClient=Mock(side_effect=[prefill, decode])))
            async with native["lifespan"](app):
                await asyncio.wait_for(app.state.ready.wait(), 1)
                self.assertEqual(await readiness(ledger.ProxyLedger(app, "unused-ledger", "mooncake")), 503)
        for data in ({}, {"1": {"engine_id": "engine-1"}}, {"0": {"engine_id": ""}}):
            with self.subTest(data=data):
                asyncio.run(run(data))

    def test_failed_native_bootstrap_never_becomes_ready(self):
        async def run():
            healthy = SimpleNamespace(raise_for_status=Mock())
            prefill = SimpleNamespace(get=AsyncMock(side_effect=[healthy, RuntimeError("bootstrap failed")]),
                                      aclose=AsyncMock())
            decode = SimpleNamespace(aclose=AsyncMock())
            native, app, _ = mooncake_proxy(SimpleNamespace(AsyncClient=Mock(side_effect=[prefill, decode])))
            tasks = []
            create_task = asyncio.create_task
            def capture(coroutine):
                task = create_task(coroutine)
                tasks.append(task)
                return task
            with patch.object(asyncio, "create_task", side_effect=capture):
                async with native["lifespan"](app):
                    with self.assertRaisesRegex(RuntimeError, "bootstrap failed"):
                        await tasks[0]
                    self.assertFalse(app.state.ready.is_set())
                    self.assertEqual(await readiness(ledger.ProxyLedger(app, "unused-ledger", "mooncake")), 503)
        asyncio.run(run())

    def test_nixl_healthcheck_is_delegated_to_native_application(self):
        app = AsyncMock()
        wrapper = ledger.ProxyLedger(app, "unused-ledger", "nixl")
        scope = {"type": "http", "method": "GET", "path": "/healthcheck"}
        receive, send = AsyncMock(), AsyncMock()
        asyncio.run(wrapper(scope, receive, send))
        app.assert_awaited_once_with(scope, receive, send)

    def test_deployment_uses_connector_appropriate_native_contract(self):
        import json
        import tempfile
        # Real deployment wiring, with Kubernetes/process/input operations mocked.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for connector in ("mooncake", "nixl"):
                with self.subTest(connector=connector):
                    directory = root / connector
                    directory.mkdir()
                    manifest = directory / "manifest.yaml"
                    manifest.write_text("mock")
                    cfg = BenchmarkConfig(backend="vllm", test_id="ready", namespace="n",
                                          target_node="prefill-node", initiator_node="decode-node",
                                          image="example/sim@sha256:" + "a" * 64, log_directory=root,
                                          parameters={"connector": connector, "nixl_backend": "mooncake" if connector == "nixl" else None})
                    kubectl = Mock()
                    kubectl.resource_exists.return_value = False
                    kubectl.output.side_effect = lambda args: "192.0.2.1" if "prefill" in " ".join(args) else "192.0.2.2"
                    runner = VllmRunner(kubectl=kubectl)
                    process = Mock(); process.poll.return_value = None
                    kubectl.start_exec.return_value = process
                    runner.server_command = Mock(return_value=["mock-server"])
                    runner._wait_ready = Mock()
                    def identity(runner, cfg, pod):
                        target = "-prefill-" in pod
                        return {"node": "prefill-node" if target else "decode-node", "pod_ip": "192.0.2.1" if target else "192.0.2.2"}
                    snapshot = {"ranks": {"dp0.tp0": {"steps": {"ok": True, "value": '{"kv_layout":{}}\n{"timing":{}}\n'}}},
                                "server_processes": {"1": {"start_ticks": 1}}}
                    with patch.object(lifecycle.inputs, "source_receipt", return_value={}), \
                         patch.object(lifecycle.inputs, "stage", return_value=({}, {})), \
                         patch.object(lifecycle, "pod_identity", side_effect=identity), \
                         patch.object(lifecycle, "capture", return_value=snapshot):
                        lifecycle.deploy(runner, cfg, manifest)
                    receipt = json.loads((directory / "servers/deployment.json").read_text())
                    self.assertEqual(receipt["status"], "ready")
                    command = receipt["proxy_command"]
                    self.assertEqual(command[2:4], ["proxy", connector])
                    url = runner._wait_ready.call_args.args[2]
                    self.assertEqual(url, receipt["proxy_readiness_url"])
                    if connector == "mooncake":
                        _, _, tree = mooncake_proxy(None)
                        routes = {(node.attr, decorator.args[0].value)
                                  for definition in tree.body if isinstance(definition, ast.AsyncFunctionDef)
                                  for decorator in definition.decorator_list if isinstance(decorator, ast.Call)
                                  and isinstance((node := decorator.func), ast.Attribute) and isinstance(node.value, ast.Name)
                                  and node.value.id == "app"}
                        self.assertEqual(routes, {("post", "/v1/completions"), ("post", "/v1/chat/completions")})
                        self.assertTrue(url.endswith("/bench/ready"))
                    else:
                        self.assertTrue(url.endswith("/healthcheck"))


if __name__ == "__main__":
    unittest.main()
