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
# Allowed outcomes: a request whose producer was killed must not count as a
# transfer; every other (not cancelled) request must have transferred.
KILLED_PRODUCER_OUTCOMES = {"failed", "fallback"}
# A request's lifecycle on one server: the simulator's kv_events records
# (real KV transfers), each server on its own clock.
# prefill_notified (decode): prefill was told to free the request's blocks
# without a read, logged under prefill's request id.
LIFECYCLE = ("reads_posted", "reads_done", "reads_failed", "finished", "sends_done",
             "prefill_notified")
# Where a cancellation landed on decode. READ_OUTSTANDING: decode's scheduler
# took the abort while the READ was in flight; ABORT_AT_READ_END: in the step
# that saw the READ end (the abort could not be taken earlier).
READ_OUTSTANDING = "READ outstanding"
ABORT_AT_READ_END = "abort taken when the READ ended"
DURING_READ = (READ_OUTSTANDING, ABORT_AT_READ_END)


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
    decoded: dict[str, int] = field(default_factory=dict)  # output tokens computed
    events: dict[str, dict[str, float]] = field(default_factory=dict)  # first t of each LIFECYCLE event
    allocations: list[tuple[float, str, list[int]]] = field(default_factory=list)
    resumed: bool = False  # a preemption happened (frees blocks without a record)
    steps: list[tuple[float, float]] = field(default_factory=list)  # (t_start, t_deadline) with tokens
    reads: list[list[float]] = field(default_factory=list)  # each NIXL read: bytes, s, s to post, descriptors
    threads: dict[int, str] = field(default_factory=dict)  # the engine process's Python threads, by native id

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
            if "kv_events" in record:
                log._add_events(record["kv_events"], float(record["t"]),
                                float(record.get("t_scheduled", record["t"])))
                continue
            if "threads" in record:
                log.threads.update({int(tid): name for tid, name in record["threads"].items()})
                continue
            if "kv_transfers" in record:
                log.reads += record["kv_transfers"].get("load", {}).get("each", [])
                continue
            if record.get("t_start") is not None and record.get("t_deadline") is not None:
                log.steps.append((float(record["t_start"]), float(record["t_deadline"])))
            for request in record.get("step", {}).get("requests", []):
                key = uuid_of(request.get("req_id"))
                if key is None:
                    continue
                log.first.setdefault(key, {**request, "t_start": record.get("t_start")})
                reached = request["num_computed_tokens"] + request["num_scheduled_tokens"]
                if reached >= request["num_prompt_tokens"]:
                    log.prefilled_at.setdefault(key, record.get("t_deadline"))
                    log.decoded[key] = max(log.decoded.get(key, 0), reached - request["num_prompt_tokens"] + 1)
        return log

    def _add_events(self, events: dict, t: float, t_scheduled: float) -> None:
        """The scheduler's events (allocations, finished requests) took effect
        by the step's start, the connector's by its end."""

        for req_id, blocks in events.get("allocated", {}).items():
            key = uuid_of(req_id)
            if key is not None:
                self.allocations.append((t_scheduled, key, list(blocks)))
                self.events.setdefault(key, {}).setdefault("allocated", t_scheduled)
        self.resumed |= bool(events.get("resumed"))
        for kind in LIFECYCLE:
            for req_id in events.get(kind, []):
                key = uuid_of(req_id)
                if key is not None:
                    at = t_scheduled if kind == "finished" else t
                    self.events.setdefault(key, {}).setdefault(kind, at)

    def released_at(self, key: str, producer: bool) -> float | None:
        """When the server could free the request's blocks: once it finished
        and, on the producer, decode read them (or their lease expired); on the
        consumer, once its READ ended. None while it still holds them."""

        events = self.events.get(key, {})
        ends = [events.get("finished")]
        if producer:
            ends.append(events.get("sends_done"))
        elif "reads_posted" in events:
            ends.append(events.get("reads_done", events.get("reads_failed")))
        return None if None in ends else max(ends)  # type: ignore[type-var]

    def handovers(self, producer: bool) -> int | None:
        """How many times a block went to a new request; raises if one went
        before its previous holder released it. None after a preemption."""

        if self.resumed:
            return None
        holder: dict[int, str] = {}
        count = 0
        for t, key, blocks in sorted(self.allocations, key=lambda allocation: allocation[0]):
            for block in blocks:
                previous = holder.get(block)
                if previous is not None and previous != key:
                    released = self.released_at(previous, producer)
                    if released is None or released > t:
                        raise ValueError(
                            f"{'prefill' if producer else 'decode'} gave block {block} to request "
                            f"{key} at t={t} while request {previous} still held it"
                        )
                    count += 1
                holder[block] = key
        return count


def classify(record: dict, prefill: StepLog, decode: StepLog, kv_check: bool,
             real: bool = False) -> dict:
    result = _classify(record, prefill, decode, kv_check, real)
    if real and "producer_killed" in record and result["outcome"] != "transferred":
        # Whatever the client saw, a dead producer must show on decode.
        result["problems"] = result.get("problems", []) + _producer_failure_problems(
            record, result, decode)
    return result


def _classify(record: dict, prefill: StepLog, decode: StepLog, kv_check: bool,
              real: bool) -> dict:
    """One request's outcome and evidence.

    ``transferred``: prefill computed the prompt, decode resumed from its KV
    with all but the last token cached (and, with ``kv_check``, the received
    bytes matched prefill's and decode resumed only after the read completed).
    ``fallback``: decode recomputed the prompt itself. ``failed``: no correct
    response. ``cancelled``: the client gave up on a request it meant to
    cancel; one that completed first is checked like any other. With ``real``
    (real KV transfers), each server's lifecycle events must show where the
    request ended and that both released its blocks. Anything contradicting
    the outcome is listed in ``problems``.
    """

    length = int(record["prompt_length"])
    key = record.get("request_id")
    result: dict = {"request_id": key, "prompt_length": length}
    for name in ("producer_killed", "producer_paused", "held", "pressure"):
        if name in record:
            result[name] = record[name] if name == "pressure" else True
    if "cancel_after_s" in record:
        result["cancel_after_s"] = record["cancel_after_s"]
        if record.get("timed_out"):
            return _cancelled(record, result, prefill, decode, kv_check, real)
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
    if real:
        read = decode.events.get(key, {}).get("reads_done")
        if read is None:
            problems.append("decode reported no completed READ")
        elif d_first.get("t_start") is not None and d_first["t_start"] < read:
            problems.append("decode resumed before its KV read completed")
        if prefill.released_at(key, producer=True) is None:
            problems.append("prefill never released the request's blocks")
    hop = record.get("prefill")
    if hop is not None:
        # The helper runs in the prefill pod: one clock with the prefill server.
        done = prefill.prefilled_at.get(key)
        result["prefill_response_after_compute_s"] = None if done is None else hop["t_received"] - done
        if done is None or hop["t_received"] < done:
            problems.append("prefill handed over its KV before its modeled computation ended")
    if "cancel_after_s" in record:
        result["completed_before_cancel"] = True
    result["outcome"] = "transferred"
    if problems:
        result["problems"] = problems
    return result


def _cancelled(record: dict, result: dict, prefill: StepLog, decode: StepLog,
               kv_check: bool, real: bool) -> dict:
    """A request the client gave up on: where decode was when the abort
    landed (``stage``, on decode's clock) and that both servers let go."""

    result["outcome"] = "cancelled"
    if not real:
        return result
    key = result["request_id"]
    events = decode.events.get(key, {})
    problems = []
    ended = events.get("reads_done", events.get("reads_failed"))
    if not events and key not in decode.first:
        # The client left before decode's server took the request: prefill
        # can only rely on its lease.
        result["stage"] = "before decode admitted it"
    elif "reads_posted" not in events:
        if "prefill_notified" in events or "finished" in events:
            result["stage"] = "before decode scheduled it"
        else:
            problems.append("decode never ended the request")
    elif "finished" not in events:
        problems.append("decode never ended the request")
    elif ended is None:
        problems.append("decode's READ never ended")
    else:
        if events["reads_posted"] < events["finished"] < ended:
            result["stage"] = READ_OUTSTANDING
        elif events["reads_posted"] < events["finished"] == ended:
            result["stage"] = ABORT_AT_READ_END
        else:
            result["stage"] = "decoding"
            result["decoded_tokens"] = decode.decoded.get(key, 0)
        if result["stage"] in DURING_READ:
            # vLLM must keep the blocks until the READ ends.
            result["read"] = "done" if "reads_done" in events else "failed"
        check = decode.kv_checks.get(key)
        if kv_check and "reads_done" in events and check is not None and not check.get("ok"):
            problems.append(f"received KV bytes differ from prefill's: {check}")
    if "prefill_notified" in events and "reads_done" in events:
        # vLLM's serving layer frees prefill's blocks again on a client
        # disconnect, as if the request had never reached the engine.
        result["duplicate_free_notification"] = True
    if prefill.released_at(key, producer=True) is None:
        problems.append(
            "prefill never released the request's blocks (raise settle_seconds past "
            "prefill's lease when decode never read them)"
        )
    paused = record.get("producer_paused")
    if paused is not None:
        if result.get("stage") not in DURING_READ:
            problems.append(
                f"the cancellation meant to land while the READ was outstanding landed "
                f"{result.get('stage', 'nowhere')}"
            )
        usage = paused.get("decode_kv_usage_before_resume")
        result["decode_kv_usage_before_resume"] = usage
        if not usage:
            problems.append("decode held no KV blocks for the aborted request's outstanding READ")
        if record.get("bystanders"):
            problems += _bystanders(record, result, decode, ended)
    if problems:
        result["problems"] = problems
    return result


def _bystanders(record: dict, result: dict, decode: StepLog, read_ended: float | None) -> list[str]:
    """Requests of decode's own (no KV transfer), sent while the paused
    producer's READ was stuck, before and after the client gave up: whether
    decode kept serving them."""

    problems, seen = [], []
    resumed = record["producer_paused"]["t_resumed"]
    for bystander in record["bystanders"]:
        first = decode.first.get(bystander.get("request_id"))
        response = bystander.get("response")
        if response is None or first is None:
            problems.append(f"decode did not serve a bystander request: {bystander.get('error')}")
            continue
        if response["choices"][0].get("token_ids") != bystander["expected_token_ids"]:
            problems.append("decode returned wrong tokens for a bystander request")
            continue
        seen.append({
            # The helper's clock (the paused prefill's pod).
            "sent_after_cancel_s": bystander["t_sent"] - record["t_received"],
            "latency_s": bystander["elapsed_s"],
            "answered_after_resume_s": bystander["t_received"] - resumed,
            # Decode's clock.
            "first_step_after_read_end_s": None if read_ended is None else first["t_start"] - read_ended,
        })
    result["bystanders"] = seen
    return problems


def _producer_failure_problems(record: dict, result: dict, decode: StepLog) -> list[str]:
    """A dead producer must end the request on decode through a failed READ,
    with an answer to the client that is either an error or (recomputed) the
    right tokens, and decode must free its blocks."""

    events = decode.events.get(result["request_id"], {})
    problems = []
    if record.get("timed_out"):
        problems.append("the client timed out: decode never answered")
    elif result["outcome"] == "failed" and "response" in record:
        problems.append(f"decode answered successfully with a wrong response: {result.get('error')}")
    if "reads_failed" not in events:
        problems.append("decode reported no failed READ for the request")
    if decode.released_at(result["request_id"], producer=False) is None:
        problems.append("decode never released the request's blocks")
    return problems


def verify(records: list[dict], prefill_steps: str, decode_steps: str,
           kv_check: bool = False, scenario: str = "proxy", real: bool = False) -> list[dict]:
    """Classify every request and check the scenario's expectations.

    HTTP success alone would not catch a failed handoff: decode then falls
    back to computing the prompt itself, so the step logs decide. Any
    ``problems`` fail the run. Raises ValueError on the first violation.
    """

    prefill, decode = StepLog.parse(prefill_steps), StepLog.parse(decode_steps)
    if real:
        prefill.handovers(producer=True)
        decode.handovers(producer=False)
    results = [classify(record, prefill, decode, kv_check, real) for record in records]
    for result in results:
        where = f"request {result['request_id']} ({result['prompt_length']} prompt tokens)"
        if result.get("problems"):
            raise ValueError(f"{where} was {result['outcome']}: {'; '.join(result['problems'])}")
        if result["outcome"] == "cancelled":
            continue
        allowed = KILLED_PRODUCER_OUTCOMES if result.get("producer_killed") else {"transferred"}
        if result["outcome"] not in allowed:
            detail = result.get("error") or f"decode cached {result.get('decode_cached_tokens')} tokens"
            raise ValueError(
                f"{where} was {result['outcome']} ({detail}); the {scenario} scenario expects "
                + " or ".join(sorted(allowed))
            )
    if scenario == "cancel" and not any(r["outcome"] == "cancelled" for r in results):
        raise ValueError("the cancel scenario cancelled no request")
    if scenario == "cancel" and real and not any(r.get("stage") in DURING_READ for r in results):
        raise ValueError("the cancel scenario cancelled no request while its READ was outstanding")
    if scenario == "producer-failure" and (
        not any(r.get("producer_killed") for r in results)
        or not any(r["outcome"] == "transferred" for r in results)
    ):
        raise ValueError(
            "the producer-failure scenario needs a transfer before the producer is "
            "killed and a request whose producer was killed"
        )
    if scenario == "pressure":
        _check_pressure(records, results, prefill)
    return results


def _check_pressure(records: list[dict], results: list[dict], prefill: StepLog) -> None:
    """The request that did not fit waited for blocks that held requests
    released only once decode came for them, then reused those blocks."""

    blocked = [(r, c) for r, c in zip(records, results) if "pressure" in r]
    held = {r["request_id"]: r for r in records if r.get("held")}
    if len(blocked) != 1 or not held:
        raise ValueError("the pressure scenario needs held requests and one that had to wait")
    record, result = blocked[0]
    pressure = record["pressure"]
    if not pressure.get("still_queued") or not (pressure.get("prefill_waiting") or 0) >= 1:
        raise ValueError(f"the extra prefill request did not wait for blocks: {pressure}")
    key = record["request_id"]
    mine = [(t, set(blocks)) for t, owner, blocks in prefill.allocations if owner == key]
    if not mine:
        raise ValueError("prefill logged no blocks for the request that waited")
    t_allocated, blocks = mine[0]
    reused = {}
    for t, owner, theirs in prefill.allocations:
        if owner in held and blocks & set(theirs):
            reused[owner] = sorted(blocks & set(theirs))
    if not reused:
        raise ValueError("the request that waited reused no block of a held request")
    for owner in reused:
        # One clock: the helper runs in the prefill pod.
        released = prefill.events.get(owner, {}).get("sends_done")
        if released is None or released < held[owner]["t_sent"]:
            raise ValueError(f"prefill released request {owner}'s blocks before decode asked for them")
    result["reused_held_blocks"] = reused
    result["allocated_after_release_s"] = t_allocated - min(
        prefill.events[owner]["sends_done"] for owner in reused)


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


def summarize(checks: list[dict], prefill_metrics: str, decode_metrics: str,
              prefill_steps: str = "", decode_steps: str = "") -> dict:
    """Outcome counts, the bytes checked, physical block reuse on decode and,
    from the lifecycle events, block handovers on each server, and both
    servers' native NIXL counters (absent when a server was down)."""

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
    stages = [c["stage"] for c in checks if c.get("stage")]
    if stages:
        summary["cancel_stages"] = {stage: stages.count(stage) for stage in sorted(set(stages))}
        summary["duplicate_free_notifications"] = sum(
            1 for c in checks if c.get("duplicate_free_notification"))
    for check in checks:
        if "bystanders" in check:
            summary["bystanders_during_stuck_read"] = check["bystanders"]
    for side, text, producer in (("prefill", prefill_steps, True), ("decode", decode_steps, False)):
        log = StepLog.parse(text)
        if log.allocations:
            summary[f"{side}_block_handovers"] = log.handovers(producer)
    for side, text in (("prefill", prefill_metrics), ("decode", decode_metrics)):
        totals = prometheus_totals(text)
        if totals:
            summary[f"{side}_metrics"] = {
                key: totals[name] for key, name in METRICS.items() if name in totals
            }
    return summary


def check_scenario(scenario: str, summary: dict, metadata: Mapping[str, object]) -> None:
    """Server-side expectations of a scenario, beyond each request's outcome:
    every live server's metrics were read and it holds no KV blocks."""

    decode = summary.get("decode_metrics", {})
    prefill = summary.get("prefill_metrics", {})
    # producer-failure kills prefill: only decode is left to check.
    live = {"decode": decode} if scenario == "producer-failure" else {"decode": decode, "prefill": prefill}
    for side, metrics in live.items():
        if "kv_cache_usage" not in metrics:
            raise ValueError(f"no KV cache usage from {side}'s metrics: cannot check its cleanup")
        if metrics["kv_cache_usage"] > 0:
            raise ValueError(
                f"{side} still holds KV blocks after the requests "
                f"({metrics['kv_cache_usage']:.1%} in use); raise settle_seconds past "
                "prefill's lease if this is prefill"
            )
    if scenario in ("proxy", "handoff", "pressure"):
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
    elif scenario == "producer-failure":
        if metadata.get("decode_healthy_after") is not True:
            raise ValueError(f"decode is not healthy after the producer failure: {metadata.get('decode_health_error')}")
        if not decode.get("nixl_failed_transfers"):
            raise ValueError(f"decode's NIXL connector counted no failed transfer: {decode}")


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
                real=options.get("kv_transfer_mode") == "real",
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
