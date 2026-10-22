from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Mapping, cast

from ...benchmark.cpu import cpu_metadata
from ...common.types import JsonValue
from ...graphing.contracts import Observation


JsonObject = Mapping[str, object]
LATENCY_METRIC = "latency"


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# The scenarios' allowed outcomes: anything else fails the run.
REQUIRED_OUTCOMES = {
    "proxy": {"transferred"},
    "handoff": {"transferred"},
    "producer-failure": {"failed", "fallback"},
}


def uuid_of(request_id: object) -> str | None:
    """The uuid shared by a request's ids on the client, prefill and decode."""

    match = _UUID.search(str(request_id or ""))
    return match.group(0) if match else None


@dataclass
class StepLog:
    """One server's simulator step log, by request uuid."""

    first: dict[str, dict] = field(default_factory=dict)  # first step: request + t_start
    prefilled_at: dict[str, float] = field(default_factory=dict)  # t_deadline of its last prefill step
    kv_checks: dict[str, dict] = field(default_factory=dict)  # decode: byte check + t

    @classmethod
    def parse(cls, text: str) -> "StepLog":
        log = cls()
        for line in text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "kv_check" in record:
                key = uuid_of(record["kv_check"].get("req_id"))
                if key is not None:
                    log.kv_checks[key] = {**record["kv_check"], "t": record.get("t")}
                continue
            for request in record.get("step", {}).get("requests", []):
                key = uuid_of(request.get("req_id"))
                if key is None:
                    continue
                log.first.setdefault(key, {**request, "t_start": record.get("t_start")})
                if request["num_computed_tokens"] + request["num_scheduled_tokens"] >= request["num_prompt_tokens"]:
                    log.prefilled_at.setdefault(key, record.get("t_deadline"))
        return log


def classify(record: dict, prefill: StepLog, decode: StepLog, kv_check: bool) -> dict:
    """One request's outcome and evidence.

    ``transferred``: prefill computed the prompt, decode resumed from its KV
    with all but the last token cached (and, with ``kv_check``, the received
    bytes matched prefill's and decode resumed only after the read completed).
    ``fallback``: decode recomputed the prompt itself. ``failed``: no correct
    response. ``cancelled``: the client dropped the request. A transfer
    problem seen anywhere is an ``error`` in a ``transferred`` request.
    """

    length = int(record["prompt_length"])
    key = record.get("request_id")
    result: dict = {"request_id": key, "prompt_length": length}
    if record.get("cancelled"):
        result["outcome"] = "cancelled"
        result["cancel_after_s"] = record.get("cancel_after_s")
        result["completed"] = "response" in record
        return result
    if "error" in record or key is None:
        result.update(outcome="failed", error=record.get("error", "no request id"))
        return result
    response = record["response"]
    choice = response["choices"][0]
    result["finish_reason"] = choice.get("finish_reason")
    if choice.get("token_ids") != record["expected_token_ids"] or choice.get("finish_reason") not in ("length", "stop"):
        result.update(outcome="failed", error=f"returned {choice.get('token_ids')} ({choice.get('finish_reason')})")
        return result
    p_first, d_first = prefill.first.get(key), decode.first.get(key)
    if d_first is None:
        result.update(outcome="failed", error="never ran on decode")
        return result
    result["decode_cached_tokens"] = int(d_first["num_computed_tokens"])
    if (d_first["num_computed_tokens"], d_first["num_scheduled_tokens"]) != (length - 1, 1):
        result["outcome"] = "fallback"
        return result
    problems = []
    if p_first is None or p_first["num_computed_tokens"] != 0:
        problems.append("prefill did not compute the prompt from scratch")
    if kv_check:
        check = decode.kv_checks.get(key)
        if check is None:
            problems.append("no KV byte check on decode")
        else:
            result["kv_check"] = {k: check.get(k) for k in ("blocks", "layers", "bytes", "mismatched_bytes", "block_ids")}
            if not check.get("ok"):
                problems.append(f"received KV bytes differ from prefill's: {check}")
            # Decode's own clock: its first step for the request comes after
            # the step that saw the read complete.
            if check.get("t") is not None and d_first.get("t_start") is not None and d_first["t_start"] < check["t"]:
                problems.append("decode resumed before its KV read completed")
    hop = record.get("prefill")
    if hop is not None:
        # The helper runs in the prefill pod: one clock with the prefill server.
        done = prefill.prefilled_at.get(key)
        result["prefill_response_after_compute_s"] = None if done is None else hop["t_received"] - done
        if done is None or hop["t_received"] < done:
            problems.append("prefill handed over its KV before its modeled computation ended")
    result["outcome"] = "transferred"
    if problems:
        result["error"] = "; ".join(problems)
    return result


def verify(records: list[dict], prefill_steps: str, decode_steps: str,
           kv_check: bool = False, scenario: str = "proxy") -> list[dict]:
    """Classify every request and check the scenario's expectations.

    HTTP success alone would not catch a failed handoff: decode then falls
    back to computing the prompt itself, so the step logs decide. A
    ``transferred`` request with an ``error`` (bytes, ordering) always fails
    the run. Raises ValueError on the first violation.
    """

    prefill, decode = StepLog.parse(prefill_steps), StepLog.parse(decode_steps)
    results = [classify(record, prefill, decode, kv_check) for record in records]
    for result in results:
        where = f"request {result['request_id']} ({result['prompt_length']} prompt tokens)"
        if result.get("outcome") == "transferred" and "error" in result:
            raise ValueError(f"{where}: {result['error']}")
        if result["outcome"] == "cancelled":
            continue
        allowed = REQUIRED_OUTCOMES.get(scenario, {"transferred"})
        if result["outcome"] not in allowed:
            detail = result.get("error") or f"decode cached {result.get('decode_cached_tokens')} tokens"
            raise ValueError(
                f"{where} was {result['outcome']} ({detail}); the {scenario} scenario expects "
                + " or ".join(sorted(allowed))
            )
    if scenario == "cancel" and not any(r["outcome"] == "cancelled" for r in results):
        raise ValueError("the cancel scenario cancelled no request")
    return results


METRICS = {
    # Native NIXL connector records (cumulative since the server started).
    "nixl_failed_transfers": "vllm:nixl_num_failed_transfers_total",
    "nixl_failed_notifications": "vllm:nixl_num_failed_notifications_total",
    "nixl_expired_requests": "vllm:nixl_num_kv_expired_reqs_total",
    "nixl_transfers": "vllm:nixl_xfer_time_seconds_count",
    "nixl_transfer_seconds": "vllm:nixl_xfer_time_seconds_sum",
    "nixl_bytes": "vllm:nixl_bytes_transferred_sum",
    "nixl_descriptors": "vllm:nixl_num_descriptors_sum",
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
}


def prometheus_totals(text: str) -> dict[str, float]:
    """Each metric's value summed over its label sets."""

    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_and_labels, _, value = line.rpartition(" ")
        name = name_and_labels.split("{", 1)[0]
        try:
            totals[name] = totals.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return totals


def summarize(checks: list[dict], prefill_metrics: str, decode_metrics: str) -> dict:
    """Outcome counts, the bytes checked, physical block reuse on decode, and
    both servers' native NIXL counters (absent when a server was down)."""

    outcomes: dict[str, int] = {}
    for check in checks:
        outcomes[check["outcome"]] = outcomes.get(check["outcome"], 0) + 1
    transferred = [c for c in checks if c["outcome"] == "transferred" and c.get("kv_check")]
    block_users: dict[int, int] = {}
    for check in transferred:
        for block in check["kv_check"].get("block_ids") or []:
            block_users[block] = block_users.get(block, 0) + 1
    summary: dict = {
        "outcomes": outcomes,
        "kv_bytes_checked": sum(int(c["kv_check"]["bytes"] or 0) for c in transferred),
        "decode_blocks_used": len(block_users),
        "decode_blocks_reused": sum(1 for users in block_users.values() if users > 1),
    }
    for side, text in (("prefill", prefill_metrics), ("decode", decode_metrics)):
        totals = prometheus_totals(text)
        if totals:
            summary[f"{side}_metrics"] = {
                key: totals[name] for key, name in METRICS.items() if name in totals
            }
    return summary


def check_scenario(scenario: str, summary: dict, metadata: Mapping[str, object]) -> None:
    """Server-side expectations of a scenario, beyond each request's outcome."""

    decode = summary.get("decode_metrics", {})
    prefill = summary.get("prefill_metrics", {})
    if scenario in ("proxy", "handoff"):
        if decode.get("nixl_failed_transfers", 0) > 0:
            raise ValueError(f"decode's NIXL connector reports failed transfers: {decode}")
        if summary["kv_bytes_checked"] and decode.get("nixl_bytes", 0) < summary["kv_bytes_checked"]:
            raise ValueError(
                f"decode's NIXL records {decode.get('nixl_bytes')} bytes, fewer than the "
                f"{summary['kv_bytes_checked']} it checked"
            )
        target = cast(Mapping[str, str], metadata.get("addresses", {})).get("target_address")
        connections = metadata.get("decode_connections")
        if target and isinstance(connections, list) and not any(
            c.split("->")[1].startswith(f"{target}:") for c in connections
        ):
            raise ValueError(f"decode has no connection to prefill's address {target}: {connections}")
    elif scenario == "cancel":
        for side, metrics in (("decode", decode), ("prefill", prefill)):
            if metrics.get("kv_cache_usage", 0) > 0:
                raise ValueError(
                    f"{side} still holds KV blocks after the cancellations "
                    f"({metrics['kv_cache_usage']:.1%} in use); raise settle_seconds past "
                    "prefill's lease if this is prefill"
                )
    elif scenario == "producer-failure":
        if metadata.get("decode_healthy_after") is not True:
            raise ValueError(f"decode is not healthy after the producer failure: {metadata.get('decode_health_error')}")


class VllmParser:
    """Parse vLLM P/D runs into per-request latency observations."""

    def discover(self, raw_directory: Path) -> tuple[Path, ...]:
        directories = tuple(sorted(
            path for path in raw_directory.glob("t*")
            if path.is_dir() and any(path.glob("*-command.json"))
        ))
        if not directories:
            raise ValueError(f"no vLLM command metadata found in {raw_directory}")
        return directories

    def parse(self, run_directory: Path) -> tuple[Observation, ...]:
        observations: list[Observation] = []
        for command_path in sorted(run_directory.glob("*-command.json")):
            metadata = _load_object(command_path)
            if metadata.get("status") != "passed":
                raise ValueError(f"vLLM run did not pass according to {command_path}")
            files = cast(Mapping[str, str], metadata["files"])
            records = json.loads((run_directory / files["responses"]).read_text())["requests"]
            options = cast(Mapping[str, object], metadata.get("options", {}))
            checked = verify(
                records,
                (run_directory / files["prefill_steps"]).read_text(),
                (run_directory / files["decode_steps"]).read_text(),
                kv_check=bool(options.get("kv_pattern_check")),
                scenario=str(options.get("scenario", "proxy")),
            )
            threads = int(cast(Mapping[str, object], metadata["parameters"])["threads"])
            run_id = f"{metadata['test_id']}-t{threads}-run{metadata['run_number']}"
            cpu = cpu_metadata(run_directory, metadata,
                               float(metadata["requests_started_s"]),  # type: ignore[arg-type]
                               float(metadata["requests_finished_s"]))  # type: ignore[arg-type]
            for record, check in zip(records, checked):
                # Latency only for requests whose KV actually moved.
                if check["outcome"] != "transferred":
                    continue
                elapsed = float(record["elapsed_s"])
                if not math.isfinite(elapsed):
                    raise ValueError(f"non-finite latency in {command_path}")
                details: dict[str, JsonValue] = {**check, **cpu}
                observations.append(Observation(
                    metric=LATENCY_METRIC,
                    value=elapsed,
                    unit="s",
                    dimensions={"threads": threads, "prompt_length": int(record["prompt_length"])},
                    sample_kind="request",
                    run_id=run_id,
                    metadata=details,
                ))
        if not observations:
            raise ValueError(f"no vLLM requests found in {run_directory}")
        return tuple(observations)


def _load_object(path: Path) -> JsonObject:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read vLLM command metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return cast(JsonObject, value)
