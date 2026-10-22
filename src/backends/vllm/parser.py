from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, cast

from ...benchmark.cpu import cpu_metadata
from ...common.types import JsonValue
from ...graphing.contracts import Observation


JsonObject = Mapping[str, object]
LATENCY_METRIC = "latency"


def first_steps(text: str) -> dict[int, dict]:
    """The first scheduled step of each request in a simulator step log, by prompt length."""

    seen: dict[str, dict] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for request in record.get("step", {}).get("requests", []):
            seen.setdefault(str(request["req_id"]), request)
    by_length: dict[int, dict] = {}
    for request in seen.values():
        length = int(request["num_prompt_tokens"])
        if length in by_length:
            raise ValueError(f"two requests with {length} prompt tokens in one step log")
        by_length[length] = request
    return by_length


def kv_checks(text: str) -> dict[str, dict]:
    """The `kv_check` records of a decode step log, by decode request id."""

    checks = {}
    for line in text.splitlines():
        if line.strip():
            record = json.loads(line)
            if "kv_check" in record:
                checks[str(record["kv_check"]["req_id"])] = record["kv_check"]
    return checks


def verify(records: list[dict], prefill_steps: str, decode_steps: str,
           kv_check: bool = False) -> list[dict]:
    """Check that every request was prefilled on P and resumed from its KV on D.

    HTTP success alone would not catch a failed handoff: decode then falls
    back to computing the prompt itself. Its first step resumes at zero in that
    case, while a remote KV load makes it start with all but the last prompt
    token cached. With ``kv_check``, decode must also have compared the bytes
    it received with prefill's pattern (`kv_check` records), and they must
    match. Raises ValueError on the first violation.
    """

    prefill = first_steps(prefill_steps)
    decode = first_steps(decode_steps)
    byte_checks = kv_checks(decode_steps) if kv_check else {}
    checked = []
    for record in records:
        length = int(record["prompt_length"])
        where = f"request with {length} prompt tokens"
        if "error" in record:
            raise ValueError(f"{where} failed: {record['error']}")
        response = record["response"]
        choice = response["choices"][0]
        if choice.get("token_ids") != record["expected_token_ids"]:
            raise ValueError(f"{where} returned {choice.get('token_ids')}, "
                             f"expected {record['expected_token_ids']}")
        if response["usage"]["prompt_tokens"] != length:
            raise ValueError(f"{where} reports {response['usage']['prompt_tokens']} prompt tokens")
        if length not in prefill:
            raise ValueError(f"{where} never ran on prefill")
        if length not in decode:
            raise ValueError(f"{where} never ran on decode")
        p_first, d_first = prefill[length], decode[length]
        if p_first["num_computed_tokens"] != 0:
            raise ValueError(f"{where} started on prefill with "
                             f"{p_first['num_computed_tokens']} cached tokens")
        if (d_first["num_computed_tokens"], d_first["num_scheduled_tokens"]) != (length - 1, 1):
            raise ValueError(
                f"{where} started on decode with {d_first['num_computed_tokens']} cached and "
                f"{d_first['num_scheduled_tokens']} scheduled tokens instead of "
                f"{length - 1} and 1: decode did not resume from prefill's KV"
            )
        check = {
            "prompt_length": length,
            "prefill_req_id": str(p_first["req_id"]),
            "decode_req_id": str(d_first["req_id"]),
            "decode_cached_tokens": int(d_first["num_computed_tokens"]),
        }
        if kv_check:
            bytes_check = byte_checks.get(str(d_first["req_id"]))
            if bytes_check is None:
                raise ValueError(f"{where} has no KV byte check in decode's step log")
            if not bytes_check.get("ok"):
                raise ValueError(f"{where}: received KV bytes do not match prefill's: {bytes_check}")
            check["kv_check"] = {k: bytes_check[k] for k in ("blocks", "layers", "bytes", "mismatched_bytes")}
        checked.append(check)
    return checked


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
            )
            threads = int(cast(Mapping[str, object], metadata["parameters"])["threads"])
            run_id = f"{metadata['test_id']}-t{threads}-run{metadata['run_number']}"
            cpu = cpu_metadata(run_directory, metadata,
                               float(metadata["requests_started_s"]),  # type: ignore[arg-type]
                               float(metadata["requests_finished_s"]))  # type: ignore[arg-type]
            for record, check in zip(records, checked):
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
