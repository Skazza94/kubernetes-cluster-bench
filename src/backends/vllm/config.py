"""Current simulator configuration, independent of Kubernetes side effects."""
from __future__ import annotations

import ipaddress
import json
import math
import re
from pathlib import Path
from typing import Mapping

from ...benchmark.contracts import BenchmarkConfig
from ...common.types import JsonValue

ROOT = Path(__file__).resolve().parents[3]
FORK = ROOT / "vllm"
NIXL_PROXY = Path("tests/v1/kv_connector/nixl_integration/toy_proxy_server.py")
MOONCAKE_PROXY = Path("examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py")
BACKENDS = {"mooncake": "Mooncake", "ucx": "UCX", "libfabric": "LIBFABRIC"}
MAPPING_OPTIONS = frozenset({"env", "bench_args", "hp_tcp_args", "kv_transfer_timing"})
DEFAULTS: dict[str, JsonValue] = {
    "workload": "dataset", "model": None, "model_revision": None, "model_path": None,
    "timing": None, "clock": "real", "kv_transfer_mode": "real",
    "connector": None, "nixl_backend": None, "kv_check": False,
    "kv_pattern_fill": False, "kv_pattern_check": False,
    "prefill_tensor_parallel_size": 1, "decode_tensor_parallel_size": 1,
    "prefill_data_parallel_size": 1, "decode_data_parallel_size": 1,
    "enable_expert_parallel": False, "enable_prefix_caching": False,
    "trust_remote_code": False, "language_model_only": False,
    "dtype": "auto", "kv_cache_dtype": "auto", "block_size": 16,
    "kv_cache_memory_bytes": 2 * 1024**3, "max_model_len": 4096,
    "max_num_batched_tokens": 2048, "max_num_seqs": 32,
    "dataset_name": "sharegpt", "dataset_path": None, "num_prompts": 100,
    "request_rate": "inf", "burstiness": 1.0, "seed": 0,
    "profile_interval_seconds": 0.1, "profile_idle_seconds": 1.0, "startup_timeout_seconds": 600,
    "request_timeout_seconds": 120, "scenario": "proxy", "prompt_lengths": "64,96",
    "output_tokens": 4, "repeat": 1, "cancel_delays": "0.001,0.01,0.05,0.2,1",
    "pause_after_cancel_seconds": 1, "settle_seconds": 0,
}
BENCH_OWN_ARGS = frozenset({
    "backend", "base-url", "host", "port", "endpoint", "model", "tokenizer", "served-model-name",
    "dataset-name", "dataset-path", "num-prompts", "max-concurrency", "request-rate",
    "burstiness", "seed", "ignore-eos", "save-result", "save-detailed", "result-dir",
    "result-filename", "trust-remote-code", "percentile-metrics", "metric-percentiles",
    "num-warmups", "ready-check-timeout-sec", "probe-request-rate",
})
MANAGED_ENV = frozenset({
    "MC_USE_TENT", "MC_FORCE_TCP", "MC_TENT_CONF", "VLLM_HOST_IP",
    "VLLM_NIXL_SIDE_CHANNEL_HOST", "VLLM_NIXL_SIDE_CHANNEL_PORT",
    "NIXL_MOONCAKE_IP_ADDR", "VLLM_MOONCAKE_BOOTSTRAP_PORT",
    "TENT_METRICS_HTTP_HOST", "TENT_METRICS_HTTP_PORT", "TENT_METRICS_REPORT_INTERVAL",
    "UCX_NET_DEVICES", "UCX_CONFIG_FILE", "FI_TCP_IFACE",
    "VLLM_SIM_TENT_METRICS_BASE_PORT",
})


def options(config: BenchmarkConfig) -> dict:
    resolved = {**DEFAULTS, **config.parameters}
    return {**resolved, **effective_pattern_flags(resolved)}


def effective_pattern_flags(*sources: Mapping[str, object]) -> dict[str, bool]:
    """Resolve the combined and individual byte-check flags.

    Any recorded source requesting checks requires them; another source's
    false/default individual flag cannot weaken that requirement.
    """
    flags = {"kv_pattern_fill": False, "kv_pattern_check": False}
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("KV pattern settings must be a mapping")
        for name in ("kv_check", *flags):
            if not isinstance(source.get(name, False), bool):
                raise ValueError(f"vLLM {name} must be true or false")
        for name in flags:
            flags[name] |= source.get("kv_check", False) or source.get(name, False)
    return flags


def scalar_mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict) or any(
        not isinstance(k, str) or not isinstance(v, (str, int, float, bool))
        or (isinstance(v, float) and not math.isfinite(v)) for k, v in value.items()
    ):
        raise ValueError(f"vLLM {name} must map names to finite scalar values")
    return value


def validate(config: BenchmarkConfig) -> dict:
    unknown = set(config.parameters) - (set(DEFAULTS) | MAPPING_OPTIONS | {"timing_args", "threads", "threads_spec", "run_number"})
    if unknown:
        raise ValueError(f"unsupported vLLM options: {', '.join(sorted(unknown))}")
    o = options(config)
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", config.image):
        raise ValueError("vLLM image must be pinned as repository@sha256:<manifest digest>")
    if o["workload"] not in ("dataset", "fixture"):
        raise ValueError("vLLM workload must be dataset or fixture")
    if o["clock"] != "real":
        raise ValueError("served KV transfers require clock real")
    timing = o["timing"]
    if not isinstance(timing, str) or timing not in ("zero", "linear", "table", "replay", "aisimulate"):
        raise ValueError("timing must be zero, linear, table, replay or aisimulate")
    args = o.get("timing_args", {})
    if not isinstance(args, dict):
        raise ValueError("vLLM timing_args must be a mapping")
    try:
        json.dumps(args, allow_nan=False)
    except (ValueError, TypeError) as e:
        raise ValueError("timing_args must contain finite JSON values") from e
    if timing in ("table", "replay"):
        path = args.get("path")
        if not isinstance(path, str) or not Path(path).is_file():
            raise ValueError("table/replay timing_args.path must name an existing recording")
    if o["connector"] not in ("nixl", "mooncake"):
        raise ValueError("explicit connector required: nixl or mooncake")
    if o["connector"] == "nixl" and o["nixl_backend"] not in BACKENDS:
        raise ValueError("explicit nixl_backend required: mooncake, ucx or libfabric")
    if o["connector"] == "mooncake" and o["nixl_backend"] is not None:
        raise ValueError("nixl_backend applies only to connector nixl")
    if o["kv_transfer_mode"] not in ("real", "simulated"):
        raise ValueError("vLLM kv_transfer_mode must be simulated or real")
    if o["workload"] == "dataset" and o["kv_transfer_mode"] != "real":
        raise ValueError("dataset KV measurements require kv_transfer_mode real")
    if o["kv_transfer_mode"] == "real" and "kv_transfer_timing" in o:
        raise ValueError("real transfers take their actual time; remove kv_transfer_timing")
    for name in ("kv_check", "kv_pattern_fill", "kv_pattern_check", "enable_expert_parallel",
                 "enable_prefix_caching", "trust_remote_code", "language_model_only"):
        if not isinstance(o[name], bool):
            raise ValueError(f"vLLM {name} must be true or false")
    if any(o[k] for k in ("kv_check", "kv_pattern_fill", "kv_pattern_check")) and o["kv_transfer_mode"] != "real":
        raise ValueError("KV pattern checks need kv_transfer_mode real")
    if o["kv_pattern_check"] and not (o["kv_pattern_fill"] or o["kv_check"]):
        raise ValueError("kv_pattern_check needs kv_pattern_fill")
    positive = ("kv_cache_memory_bytes", "max_model_len", "max_num_batched_tokens", "max_num_seqs",
                "num_prompts", "output_tokens", "repeat", "startup_timeout_seconds", "request_timeout_seconds",
                "prefill_tensor_parallel_size", "decode_tensor_parallel_size",
                "prefill_data_parallel_size", "decode_data_parallel_size")
    for name in positive:
        if type(o[name]) is not int or o[name] < 1:
            raise ValueError(f"vLLM {name} must be a positive integer")
    if o["block_size"] is not None and (type(o["block_size"]) is not int or o["block_size"] < 16 or o["block_size"] % 16):
        raise ValueError("vLLM block_size must be a positive multiple of 16, or null for platform geometry")
    for side in ("prefill", "decode"):
        if o[f"{side}_data_parallel_size"] > 1 and not o["enable_expert_parallel"]:
            raise ValueError("real DP requires enable_expert_parallel and an MoE model")
    if o["workload"] == "fixture" and (o["connector"] != "nixl" or any(
        o[f"{side}_{parallel}_parallel_size"] != 1 for side in ("prefill", "decode") for parallel in ("tensor", "data")
    )):
        raise ValueError("lifecycle fixtures require NIXL at TP=DP=1")
    for name in ("profile_interval_seconds", "profile_idle_seconds", "pause_after_cancel_seconds", "settle_seconds"):
        v = o[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"vLLM {name} must be finite and non-negative")
    for name in MAPPING_OPTIONS:
        if name in o:
            scalar_mapping(o[name], name)
    env = {k: str(v) for k, v in o.get("env", {}).items()}
    if any(not re.fullmatch(r"[A-Z_][A-Z0-9_]*", k) for k in env):
        raise ValueError("env requires valid environment variable names")
    if any(re.search(r"TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY|PRIVATE_KEY", k) for k in env):
        raise ValueError("credentials belong in host loader environment, not recorded profile env")
    if set(env) & MANAGED_ENV or any((k.startswith("NIXL_") and k != "NIXL_LOG_LEVEL") or k.startswith("TENT_") for k in env):
        raise ValueError("managed transport identity/binding cannot be overridden in env (including MC_FORCE_TCP)")
    non_tent = o["connector"] == "nixl" and o["nixl_backend"] != "mooncake"
    if non_tent and (o.get("hp_tcp_args") or any(k.startswith("MC_") for k in env)):
        raise ValueError("UCX/LIBFABRIC does not accept TENT/Mooncake settings")
    if o["nixl_backend"] == "ucx" and "UCX_TLS" in env and env["UCX_TLS"] != "tcp,self":
        raise ValueError("managed UCX requires UCX_TLS=tcp,self")
    if o["nixl_backend"] == "libfabric":
        policy = {"FI_PROVIDER": "tcp", "FI_ADDR_FORMAT": "FI_SOCKADDR_IN", "FI_PREFER_SYSCONFIG": "0"}
        diagnostics = {"FI_LOG_LEVEL", "FI_LOG_PROV", "FI_LOG_SUBSYS"}
        if any(k.startswith("FI_") and (k not in set(policy) | diagnostics or k in policy and env[k] != policy[k]) for k in env):
            raise ValueError("managed LIBFABRIC requires direct TCP/IPv4 policy; FI overrides unsupported")
    elif any(k.startswith("FI_") for k in env):
        raise ValueError("FI settings apply only to LIBFABRIC")
    if o["nixl_backend"] != "ucx" and any(k.startswith("UCX_") for k in env):
        raise ValueError("UCX settings apply only to UCX")
    hp = o.get("hp_tcp_args", {})
    limits = {"max_outstanding_tasks", "max_outstanding_bytes", "max_transfer_bytes"}
    if hp.keys() - limits or any(type(v) is not int or v < 1 for v in hp.values()):
        raise ValueError("hp_tcp_args supports positive integer max_outstanding_tasks/max_outstanding_bytes/max_transfer_bytes only")
    if o["workload"] == "dataset":
        fixture_only = {"scenario", "prompt_lengths", "output_tokens", "repeat", "cancel_delays", "pause_after_cancel_seconds"}
        if fixture_only & config.parameters.keys():
            raise ValueError("dataset workloads use bench_args, not fixture scenario/prompt/output options")
        if not isinstance(o["model"], str) or not o["model"]:
            raise ValueError("dataset workloads require model")
        if not isinstance(o["model_revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", o["model_revision"]):
            raise ValueError("model_revision must be an explicit 40-character commit")
        if o["model_path"] is not None and (not isinstance(o["model_path"], str) or not (Path(o["model_path"]) / "config.json").is_file()):
            raise ValueError("model_path must contain config.json")
        if not isinstance(o["dataset_name"], str) or not o["dataset_name"]:
            raise ValueError("dataset_name must name a vllm bench serve dataset")
        if o["dataset_path"] is not None and not isinstance(o["dataset_path"], str):
            raise ValueError("dataset_path must be a file, URL or HF dataset id")
        if o["dataset_name"] == "hf" and not o["dataset_path"]:
            raise ValueError("dataset_name hf requires dataset_path")
        rate = o["request_rate"]
        if rate != "inf" and (isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0):
            raise ValueError("request_rate must be inf or a finite positive number")
        if type(o["seed"]) is not int or o["seed"] < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(o["burstiness"], bool) or not isinstance(o["burstiness"], (int, float)) or not math.isfinite(o["burstiness"]) or o["burstiness"] <= 0:
            raise ValueError("burstiness must be finite and positive")
        bench = o.get("bench_args", {})
        if BENCH_OWN_ARGS & bench.keys() or any(not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9-]*", k) for k in bench):
            raise ValueError("bench_args must use unowned option names without '--'")
    return o


def server_settings(config: BenchmarkConfig, address: str, role: str, timing_path: str | None = None) -> tuple[dict, dict, dict]:
    o = validate(config)
    if role not in ("initiator", "target"):
        raise ValueError("server role must be initiator (decode) or target (prefill)")
    side = "decode" if role == "initiator" else "prefill"
    tp, dp = o[f"{side}_tensor_parallel_size"], o[f"{side}_data_parallel_size"]
    ipaddress.IPv4Address(address)
    args = dict(o.get("timing_args", {}))
    if timing_path is not None:
        args["path"] = timing_path
    sim = {"clock": o["clock"], "timing": o["timing"], "timing_args": args, "log_path": "/tmp/sim-steps.jsonl"}
    real = o["kv_transfer_mode"] == "real"
    if real:
        sim["kv_transfer_mode"] = "real"
        pattern = "kv_pattern_fill" if side == "prefill" else "kv_pattern_check"
        if o[pattern]:
            sim[pattern] = True
    else:
        sim["kv_transfer_timing"] = o.get("kv_transfer_timing", {"load_gbps": 1, "load_latency_us": 50000})
    nixl = o["connector"] == "nixl"
    extra = {"backends": [BACKENDS[o["nixl_backend"]]]} if nixl else {"mooncake_protocol": "tcp"}
    if real:
        extra["rank_addresses"] = [address] * (tp * dp)
    kv = {"kv_connector": "NixlConnector" if nixl else "MooncakeConnector",
          "kv_role": "kv_consumer" if side == "decode" else "kv_producer",
          "kv_connector_extra_config": extra}
    if real:
        kv["kv_buffer_device"] = "cpu"
    env = {"TRITON_CPU_BACKEND": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "OMP_NUM_THREADS": "1", "VLLM_HOST_IP": address}
    if nixl:
        env.update(VLLM_NIXL_SIDE_CHANNEL_HOST=address, VLLM_NIXL_SIDE_CHANNEL_PORT="5600", VLLM_SSM_CONV_STATE_LAYOUT="DS")
    else:
        env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = "8998"
    if real and (not nixl or o["nixl_backend"] == "mooncake"):
        transports = {k: {"enable": False} for k in ("tcp", "rdma", "shm", "gds", "ascend_direct")}
        transports["hp_tcp"] = {"enable": True, "bind_address": address, "advertise_address": address,
                                "rail_addresses": [address], **o.get("hp_tcp_args", {})}
        env.update(MC_USE_TENT="1", MC_TENT_CONF=json.dumps({"rpc_server_hostname": address, "transports": transports}, separators=(",", ":")),
                   TENT_METRICS_HTTP_HOST="127.0.0.1", TENT_METRICS_HTTP_PORT="9100", TENT_METRICS_REPORT_INTERVAL="0")
        if nixl:
            env["NIXL_MOONCAKE_IP_ADDR"] = address
    elif o["nixl_backend"] == "ucx":
        env["UCX_TLS"] = "tcp,self"
    elif o["nixl_backend"] == "libfabric":
        env.update(FI_PROVIDER="tcp", FI_ADDR_FORMAT="FI_SOCKADDR_IN", FI_PREFER_SYSCONFIG="0")
    env.update({k: str(v) for k, v in o.get("env", {}).items()})
    return sim, kv, env
