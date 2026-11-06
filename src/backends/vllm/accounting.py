"""Rank-complete, regenerable accounting over preserved simulator evidence.

Payload is counted only at the mover (NIXL decode READ / Mooncake prefill
WRITE). Receiving checks, sender bookkeeping and interface counters corroborate
that payload; they are not additional payload. All intervals remain node-local.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

from . import reference_accounting as reference
from .config import effective_pattern_flags
from .ledger import join_client


UUID = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f])")
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([^\s]+)(?:\s+\d+)?$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"(?:,|$)')


def records(text):
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL line {number} is not an object")
        result.append(row)
    return result


def request_uuid(value):
    found = UUID.findall(value)
    if len(found) != 1:
        raise ValueError(f"unrecognized or ambiguous server request ID: {value}")
    return found[0]


def series(text):
    """Preserve each label set; never hide missing/negative engines in a sum."""
    result = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = SAMPLE.fullmatch(line)
        if not match:
            raise ValueError(f"invalid Prometheus sample: {line[:160]}")
        name, raw, value = match.groups()
        labels = {}
        if raw:
            body, position = raw[1:-1], 0
            for label in LABEL.finditer(body):
                if label.start() != position or label[1] in labels:
                    raise ValueError("malformed or duplicate Prometheus label")
                labels[label[1]] = json.loads('"' + label[2] + '"')
                position = label.end()
            if position != len(body):
                raise ValueError("malformed Prometheus labels")
        key = (name, tuple(sorted(labels.items())))
        number = float(value)
        if key in result or not math.isfinite(number) or number < 0:
            raise ValueError(f"duplicate/nonfinite/negative metric: {key}")
        result[key] = number
    if not result:
        raise ValueError("empty metrics scrape")
    return result


def counter_delta(before, after, name, *, allow_new=False):
    left = {key: value for key, value in series(before).items() if key[0] == name}
    right = {key: value for key, value in series(after).items() if key[0] == name}
    if not right or set(left) - set(right) or (not allow_new and set(left) != set(right)):
        raise ValueError(f"missing/changed counter series: {name}")
    delta = {key: value - left.get(key, 0) for key, value in right.items()}
    if any(v < 0 for v in delta.values()):
        raise ValueError(f"counter reset: {name}")
    return sum(delta.values())


def inventory(run):
    return {(role, 0, dp, tp) for role in ("prefill", "decode")
            for dp, tp in getattr(run, role).ranks(getattr(run, role).tp(run.server))}


def core(bench, logs, run, tent=None, routers=None, ports=None):
    """Validate rank inventory, connector telemetry and byte classification.

    Full run validation also requires client ownership, counter epochs and
    cleanup evidence.
    """
    if set(logs) != inventory(run):
        raise ValueError(f"rank inventory differs: missing={inventory(run)-set(logs)}, extra={set(logs)-inventory(run)}")
    for key, text in logs.items():
        rows = records(text)
        layouts = [r["kv_layout"] for r in rows if "kv_layout" in r]
        if len(layouts) != 1:
            raise ValueError(f"{key}: expected exactly one startup KV layout")
        layout = layouts[0]
        role, instance, dp, tp = key
        if (layout.get("dp_rank"), layout.get("tp_rank"), layout.get("tp_size")) != (
                dp, tp, getattr(run, role).tp(run.server)):
            raise ValueError(f"{key}: wrong rank identity")
    sides = {role: [logs[(role, 0, dp, tp)] for dp, tp in getattr(run, role).ranks(getattr(run, role).tp(run.server))]
             for role in ("prefill", "decode")}
    result = reference.check(bench, sides["prefill"], sides["decode"], routers, run, tent, ports)
    mover, direction = ("decode", "load") if run.server.connector == "nixl" else ("prefill", "send")
    totals = {"count": 0, "bytes": 0, "failed": 0}
    operations = []
    for key, text in logs.items():
        for row in records(text):
            for kind, batch in (row.get("kv_transfers") or {}).items():
                entries = batch.get("each", [])
                if key[0] != mover or kind != direction:
                    if batch.get("bytes") or batch.get("count") or batch.get("failed"):
                        result["problems"].append(f"{key}: unexpected non-mover telemetry {kind}")
                    continue
                for name in totals:
                    if name not in batch and name != "failed":
                        raise ValueError(f"{key}: missing transfer {name}")
                    totals[name] += int(batch.get(name, 0))
                if int(batch["count"]) != len(entries) or sum(e[0] for e in entries) != batch["bytes"]:
                    result["problems"].append(f"{key}: incomplete per-operation telemetry")
                for entry in entries:
                    operations.append({"role": key[0], "instance": key[1], "dp": key[2], "tp": key[3],
                                       "batch_t": row.get("t"), "entry": entry})
                    if run.server.connector == "nixl" and entry[0] > 0:
                        plan = entry[6]
                        if not isinstance(plan, list) or len(plan) != 5 or not isinstance(plan[4], dict):
                            result["problems"].append(f"{key}: missing selected-byte/length-bin evidence")
                        elif (sum(int(n)*count for n, count in plan[4].items()) != entry[0]
                              or sum(plan[4].values()) != plan[1] or plan[3] != entry[0]
                              or not 0 < plan[2] == entry[3] <= plan[1]
                              or (reference.uses_tent(run.server) and plan[1] > 1024)):
                            result["problems"].append(f"{key}: descriptor counts/bytes/guard disagree")
    result["physical_transfers"] = totals
    result["payload_operations"] = sum(operation["entry"][0] > 0 for operation in operations)
    result["zero_payload_operations"] = sum(operation["entry"][0] == 0 for operation in operations)
    result["operations"] = operations
    if totals != result["transfers"]:
        result["problems"].append("reference endpoint sums differ from mover-only accounting")
    try:
        result["bytes"] = classify(logs, run, totals["bytes"])
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        result["bytes"] = {"available": False, "reason": str(error), "physical_payload_bytes": totals["bytes"]}
        result["problems"].append(f"useful/replicated/padding classification unavailable: {error}")
    result["evidence_scope"] = "connector/layout only; does not certify live lifecycle or network"
    return result


def classify(logs, run, physical):
    """Canonical useful spans, replica useful copies, moved and unmoved padding.

    Attention is unique by request/group/global-head/token range, including the
    group's layers in its per-token size. State is unique by request/group/
    component/global-column range, including real layers/regions. Tail/page
    padding in *all* physical copies is padding; replicated_useful_bytes is
    redundant useful content, not redundant padding. No blanket TP division.
    """
    spans = defaultdict(list)
    useful_copies = Fraction(0)
    padding = transfer_only = unmoved_padding = accounted = 0
    head_sources = set()
    for (role, instance, dp, tp), text in logs.items():
        if role != "decode":
            continue
        rows = records(text)
        layout = next(r["kv_layout"] for r in rows if "kv_layout" in r)
        groups = layout.get("groups")
        if not groups:
            raise ValueError("current grouped KV layout required for useful-byte classification")
        plans = reference._pull_plans(rows)
        # A single full-attention group uses a flat allocation table.
        # Resolve it at reads_posted, before later scheduler allocations grow.
        posted = defaultdict(int)
        for row in rows:
            events = row.get("kv_events") or {}
            for request in events.get("reads_posted") or []:
                posted[request] += 1
                if len(groups) == 1 and not plans.get(request):
                    allocated = events.get("allocated", {}).get(request)
                    if allocated is None:
                        raise ValueError("single-group pull lacks its posted allocation")
                    plans[request] = {0: (0, sum(bool(block) for block in allocated))}
        if any(n != 1 for n in posted.values()):
            raise ValueError("repeated pull/retry for a request")
        checks = {c["req_id"]: c for r in rows if (c := r.get("kv_check")) and c.get("kind") in (None, "transfer")}
        joint = {}
        for (side, _, _, rank), source in logs.items():
            if side == "decode" and rank == tp:
                joint.update(reference._request_steps(records(source)))
        for request, plan in plans.items():
            prompt = joint[request][0][2]
            tokens = prompt - int(reference._prefill_backoff(layout) and prompt > 1)
            nested = {g["group"]: g for g in checks.get(request, {}).get("groups", [])}
            if run.kv_check and set(plan) != set(nested):
                # A skipped 1-token window may have a zero-length checked group.
                if set(plan) != {g for g, c in nested.items() if c["blocks"]}:
                    raise ValueError("check and posted pull groups differ")
            for index, (first, blocks) in plan.items():
                spec = groups[index]
                transfer = spec.get("transfer", {})
                logical_block = (transfer.get("bytes_per_block", 0) - transfer.get("transfer_only_bytes_per_block", 0)
                                 if transfer else spec.get("checked_bytes_per_block", spec["layers"] * spec.get("real_page_bytes", spec["block_bytes"])))
                moved_block = (transfer.get("bytes_per_block") if transfer else
                               spec.get("transfer_bytes_per_block", spec["layers"] * spec["block_bytes"]))
                only = blocks * transfer.get("transfer_only_bytes_per_block", 0)
                moved = blocks * moved_block
                # NIXL state padding is explicitly *not* transferred.
                unmoved_padding += blocks * transfer.get("padding_bytes_per_block", 0)
                group_useful = Fraction(0)
                base = (request_uuid(request), index)
                if spec.get("kind") == "MambaSpec":
                    state = spec["state"]
                    local = reference._state_spans(state, tp, run.decode.tp(run.server))
                    if not local:
                        raise ValueError("state spans not divisible over TP")
                    whole_local = sum(s[3] for s in local)
                    multiplier = Fraction(logical_block * blocks, whole_local)
                    for kind, component, offset, length, start, count in local:
                        unit = Fraction(length, count) * multiplier
                        spans[base + (kind, component)].append((start, start+count, unit))
                    group_useful = logical_block * blocks
                else:
                    if "members" in spec:
                        raise ValueError("packed members need per-member token-range classification")
                    heads, source = attention_heads(spec, layout, run, tp)
                    head_sources.add(source)
                    block_size = spec["block_size"]
                    lo, hi = first * block_size, min(tokens, (first+blocks) * block_size)
                    if hi < lo:
                        raise ValueError("pull lies outside handoff token range")
                    unit = Fraction(logical_block, block_size * len(heads))
                    for head in heads:
                        spans[base + ("head", head)].append((lo, hi, unit))
                    group_useful = (hi-lo) * unit * len(heads)
                page_and_tail = moved - only - group_useful
                if page_and_tail < 0:
                    raise ValueError("logical useful content exceeds physical group payload")
                useful_copies += group_useful
                padding += page_and_tail
                transfer_only += only
                accounted += moved
    if accounted != physical:
        raise ValueError(f"group-selected payload {accounted} != mover payload {physical}")
    unique = Fraction(0)
    for key, ranges in spans.items():
        points = sorted({p for lo, hi, unit in ranges for p in (lo, hi)})
        for lo, hi in zip(points, points[1:]):
            units = {unit for start, end, unit in ranges if start <= lo and hi <= end}
            if len(units) > 1:
                raise ValueError(f"inconsistent replica geometry: {key}")
            if units:
                unique += (hi-lo) * units.pop()
    values = {"useful_kv_bytes": unique, "replicated_useful_bytes": useful_copies-unique,
              "transferred_padding_bytes": padding, "transfer_only_bytes": transfer_only,
              "untransferred_page_padding_bytes": unmoved_padding}
    if any(Fraction(v).denominator != 1 or v < 0 for v in values.values()):
        raise ValueError("nonintegral/negative byte classification")
    return {"available": True, "physical_payload_bytes": physical, **{k: int(v) for k, v in values.items()},
            "head_identity_sources": sorted(head_sources),
            "identity": "physical = useful + replicated_useful + transferred_padding + transfer_only",
            "useful_definition": "unique token/head/layer and logical state spans at handoff; excludes all physical padding"}


def attention_heads(spec, layout, run, tp):
    """Unpatterned dense layouts omit head labels; derive only proven geometry.

    This does not add labels to raw records or bypass native compatibility
    checks. Unknown layouts and page widths fail instead of guessing heads.
    """
    if spec.get("global_heads"):
        return spec["global_heads"], "emitted_global_heads"
    if run.kv_check:
        raise ValueError("checked attention layout lacks global head identity")
    supported = ("FullAttentionSpec", "SlidingWindowSpec", "MLAAttentionSpec")
    if spec.get("kind") not in supported or spec.get("pattern") != "rank-v0":
        raise ValueError("unpatterned attention kind is not supported for head derivation")
    model = json.loads((run.model / "config.json").read_text())
    text = model.get("text_config") or model.get("llm_config") or model
    total, error = reference.group_kv_heads(run.model, [spec["kind"]])
    if error or not total or total[0] < 1:
        raise ValueError(f"cannot derive global heads: {error}")
    count = total[0]
    size = run.decode.tp(run.server)
    if max(count, size) % min(count, size):
        raise ValueError("unpatterned KV heads do not divide/replicate across TP")
    heads = reference.global_heads(tp, size, count)
    itemsize = {"bfloat16": 2, "float16": 2, "float32": 4, "float8_e4m3fn": 1, "float8_e5m2": 1}.get(layout.get("dtype"))
    if itemsize is None:
        raise ValueError("unpatterned KV dtype has no proven item size")
    if spec["kind"] == "MLAAttentionSpec":
        width = text["kv_lora_rank"] + text["qk_rope_head_dim"]
        expected = spec["block_size"] * itemsize * width
    else:
        width = text.get("head_dim")
        if width is None:
            if text["hidden_size"] % text["num_attention_heads"]:
                raise ValueError("model head width is not integral")
            width = text["hidden_size"] // text["num_attention_heads"]
        expected = spec["block_size"] * len(heads) * width * 2 * itemsize
    if expected != spec["real_page_bytes"] or expected > spec["block_bytes"]:
        raise ValueError("unpatterned page geometry differs from model head sharding")
    return heads, "derived_from_model_config_and_emitted_page_geometry"


def cleanup_metrics(text, dp_size):
    data = series(text)
    for name in ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting"):
        found = [(dict(labels), value) for (metric, labels), value in data.items() if metric == name]
        if len(found) != dp_size or any(value != 0 for _, value in found):
            raise ValueError(f"cleanup: {name} must explicitly be zero on every DP engine")
        engines = [labels.get("engine") for labels, _ in found]
        if any(engine is None for engine in engines) or len(set(engines)) != dp_size:
            raise ValueError(f"cleanup: {name} lacks distinct engine labels")
        if dp_size > 1 and set(engines) != {str(dp) for dp in range(dp_size)}:
            raise ValueError(f"cleanup: unexpected DP engines {engines}")


NIXL_COUNTERS = ("vllm:nixl_bytes_transferred_sum", "vllm:nixl_xfer_time_seconds_count",
                 "vllm:nixl_num_descriptors_sum", "vllm:nixl_num_failed_transfers_total",
                 "vllm:nixl_num_failed_notifications_total", "vllm:nixl_num_kv_expired_reqs_total",
                 "vllm:nixl_num_notifications_after_expiry_total")
MOONCAKE_COUNTERS = ("vllm:mooncake_bytes_transferred_sum", "vllm:mooncake_xfer_time_seconds_count",
                     "vllm:mooncake_num_descriptors_sum", "vllm:mooncake_num_failed_transfers_total",
                     "vllm:mooncake_num_failed_recvs_total", "vllm:mooncake_num_kv_expired_reqs_total")


def engine_deltas(before, after, dp_size, names=NIXL_COUNTERS):
    for text in (before, after):
        parsed = series(text)
        for name in names:
            labels = [dict(label) for metric, label in parsed if metric == name]
            if len(labels) != dp_size or {l.get("engine") for l in labels} != {str(i) for i in range(dp_size)}:
                raise ValueError(f"required per-engine counter missing: {name}")
    result = {name: counter_delta(before, after, name) for name in names}
    if any(result[name] for name in names[3:]):
        raise ValueError("connector server counters report failure, expiry or late notification")
    return result


def registered_pool(text, expected_bytes):
    values = [value for (name, labels), value in series(text).items() if name == "tent_registered_buffer_bytes"]
    if not values or sum(values) < expected_bytes:
        raise ValueError("TENT scrape lacks the expected registered KV pool")


def require(item, label):
    if item.get("ok") is not True:
        raise ValueError(f"missing required {label}: {item.get('error', item)}")
    return item["value"]


def run_spec(options, model):
    fields = reference.Server.__dataclass_fields__
    server = reference.Server(**{k: v for k, v in options.items() if k in fields})
    server.nixl_backend = options.get("nixl_backend") or "mooncake"
    sides = {role: reference.Side(role, "shared-kv", options[f"{role}_data_parallel_size"],
                                 options[f"{role}_tensor_parallel_size"]) for role in ("prefill", "decode")}
    return reference.Run("kubernetes", "existing-kubernetes", model, **sides, client="prefill", control_network="control",
                         kv_check=effective_pattern_flags(options)["kv_pattern_check"], server=server,
                         workload=reference.Workload(num_prompts=options["num_prompts"]))


def owned_ids(rows, prefix=False):
    field = "admitted" if prefix else "allocated"
    return {request for r in rows for request in (r.get("kv_events") or {}).get(field, {})}


def lifecycle_problems(logs, run, expected):
    problems = []
    steps = defaultdict(dict)
    for (role, instance, dp, tp), text in logs.items():
        for request, rows in reference._request_steps(records(text)).items():
            key = request_uuid(request)
            if key in steps[(role, tp)]:
                problems.append(f"{role}/tp{tp}: duplicate request steps across engines")
            steps[(role, tp)][key] = rows
        rows = records(text)
        allocated = owned_ids(rows, run.server.enable_prefix_caching)
        uuids = [request_uuid(request) for request in allocated]
        if len(set(uuids)) != len(uuids):
            problems.append(f"{role}/dp{dp}.tp{tp}: multiple attempts for one proxy request")
        kinds = ("reads_posted", "reads_done", "finished") if role == "decode" else ("finished", "sends_done")
        for kind in kinds:
            counts = defaultdict(int)
            for row in rows:
                for request in (row.get("kv_events") or {}).get(kind, []):
                    counts[request_uuid(request)] += 1
            needed = set(uuids)
            if role == "decode" and run.server.enable_prefix_caching and kind.startswith("reads_"):
                # The reference's checked admissions/hits prove any no-pull
                # request. Prefix hits do not create physical operations.
                needed = {request_uuid(req) for r in rows for req in (r.get("kv_events") or {}).get("reads_posted", [])}
            if set(counts) != needed or any(n != 1 for n in counts.values()):
                problems.append(f"{role}/dp{dp}.tp{tp}: {kind} missing, repeated or unowned")
        if any((r.get("kv_events") or {}).get("reads_failed") for r in rows):
            problems.append(f"{role}/dp{dp}.tp{tp}: failed KV read")
    for tp in range(run.decode.tp(run.server)):
        decoded = steps[("decode", tp)]
        if set(decoded) != expected:
            problems.append(f"decode/tp{tp}: missing/extra scheduler requests")
        for request, rows in decoded.items():
            computed, scheduled, prompt, drafts = rows[0]
            if computed < prompt-1 or scheduled > 1+drafts:
                problems.append(f"decode/tp{tp}: {request} locally prefilled instead of resuming KV")
    for tp in range(run.prefill.tp(run.server)):
        produced = steps[("prefill", tp)]
        if set(produced) != expected:
            problems.append(f"prefill/tp{tp}: missing/extra scheduler requests")
        layout = next(r["kv_layout"] for r in records(logs[("prefill", 0, 0, tp)]) if "kv_layout" in r)
        for request, rows in produced.items():
            decoded = steps[("decode", 0)].get(request)
            if not decoded:
                problems.append(f"prefill/tp{tp}: request has no decode prompt")
                continue
            prompt = decoded[0][2]
            needed = prompt - int(reference._prefill_backoff(layout) and prompt > 1)
            if any(row[2] != needed for row in rows) or max(row[0]+row[1] for row in rows) < needed:
                problems.append(f"prefill/tp{tp}: {request} did not produce its handoff prompt")
    return problems


def regenerate(directory: Path):
    """Regenerate exclusively from retained artifacts, never from live pods."""
    manifest = json.loads((directory / "manifest.json").read_text())
    problems = list(manifest.get("collection_errors", []))
    result = {"status": "failed", "problems": problems}
    try:
        run = run_spec(manifest["options"], directory / "inputs" / "model")
        expected_inventory = {tuple(r[k] for k in ("role", "instance", "dp", "tp")) for r in manifest["expected_ranks"]}
        if manifest["expected_instances"] != {"prefill": 1, "decode": 1} or expected_inventory != inventory(run):
            raise ValueError("expected instance/rank inventory differs from resolved configuration")
        receipt = json.loads((directory / "inputs" / "receipt.json").read_text())
        if not manifest["artifact_hashes"]:
            raise ValueError("empty artifact hash inventory")
        for relative, expected in manifest["artifact_hashes"].items():
            path = directory / relative
            if path.resolve().is_relative_to(directory.resolve()) is False or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"artifact hash mismatch: {relative}")
        for name, digest in receipt["model"]["sha256"].items():
            if hashlib.sha256((directory/"inputs"/"model"/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"retained model hash mismatch: {name}")
        for item in (receipt["model"], receipt.get("dataset", {}), receipt.get("timing", {})):
            if item.get("artifact"):
                digest = item.get("archive_sha256") or item["sha256"]
                if hashlib.sha256((directory/"inputs"/item["artifact"]).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"retained input differs from upload receipt: {item['artifact']}")
        client = json.loads((directory / "client.json").read_text())
        if client["exit_code"] != 0 or not client.get("bench"):
            raise ValueError("client failed or detailed result missing")
        bench = client["bench"]
        ledger = join_client(bench, records((directory / "client-ledger.jsonl").read_text()))
        if len(ledger) != run.workload.num_prompts or not all(r["success"] and not r["error"] for r in ledger):
            raise ValueError("client requests incomplete/failed")
        proxy = records((directory / "proxy-ledger.jsonl").read_text())
        mapped = [r for r in proxy if r.get("event") == "mapped"]
        finished = [r for r in proxy if r.get("event") == "finished"]
        client_ids = {r["client_id"] for r in ledger}
        if (len(mapped) != len(ledger) or {r.get("client_id") for r in mapped} != client_ids
                or len({r["proxy_id"] for r in mapped}) != len(ledger)
                or len(finished) != len(ledger) or {r.get("client_id") for r in finished} != client_ids
                or any(r.get("status") != 200 for r in finished)):
            raise ValueError("proxy/client ownership missing, ambiguous or failed")
        expected_requests = {r["proxy_id"] for r in mapped}
        logs, tent = {}, {"prefill": [], "decode": []}
        backend_evidence = {}
        server_counters = {}
        network = {}
        deployment = json.loads((directory / "deployment.json").read_text())
        requested_image = manifest["image_requested"]
        if deployment["status"] != "ready" or deployment["image_requested"] != requested_image:
            raise ValueError("deployment image/status differs from the requested run")
        if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", requested_image):
            raise ValueError("missing pinned image identity")
        for role in ("prefill", "decode"):
            original = deployment["pods"]["target" if role == "prefill" else "initiator"]
            for phase in ("before", "after"):
                pod = json.loads((directory / f"{role}-pod-{phase}.json").read_text())
                if (pod != original or not pod["uid"] or not pod["container_id"]
                        or pod["node"] != manifest["expected_nodes"][role]
                        or pod["image_requested"] != requested_image
                        or pod["image_id"].split("@")[-1] != requested_image.split("@")[-1]):
                    raise ValueError(f"{role}: pod/node/runtime image epoch differs")
            expected_hashes = {"/model/"+name: digest for name, digest in receipt["model"]["sha256"].items()}
            for kind in ("timing", "dataset"):
                item = receipt.get(kind, {})
                if item.get("sha256") and (kind != "dataset" or role == "prefill"):
                    expected_hashes[item["pod_path"]] = item["sha256"]
            if json.loads((directory / f"{role}-input-hashes.json").read_text()) != expected_hashes:
                raise ValueError(f"{role}: pod-side input hash evidence differs from captured receipt")
            before = json.loads((directory / f"{role}-before.json").read_text())
            after = json.loads((directory / f"{role}-after.json").read_text())
            if (not before["server_processes"] or before["boot_id"] != after["boot_id"]
                    or before["server_processes"] != after["server_processes"]):
                raise ValueError(f"{role}: server counter epoch changed or missing")
            if not (before["started"] <= before["finished"] <= after["started"] <= after["finished"]):
                raise ValueError(f"{role}: invalid local measurement boundaries")
            dependencies = ["vllm", "torch"]
            if run.server.connector == "nixl":
                dependencies.append("nixl")
            if run.server.timing == "aisimulate":
                dependencies.append("aisimulate")
            if run.server.nixl_backend in ("ucx", "libfabric"):
                dependencies.append(run.server.nixl_backend)
            for name in dependencies:
                version = require(before["versions"][name], f"{role} {name} version")
                if not version or version != require(after["versions"][name], f"{role} final {name} version"):
                    raise ValueError(f"{role}: dependency version changed or empty: {name}")
            if not before.get("native_binaries_sha256") or before["native_binaries_sha256"] != after.get("native_binaries_sha256"):
                raise ValueError(f"{role}: native binary identities missing/changed")
            for name, value in before["native_binaries_sha256"].items():
                require(value, f"{role} native binary {name}")
            side = getattr(run, role)
            expected_rank_keys = {f"dp{dp}.tp{tp}" for dp, tp in side.ranks(side.tp(run.server))}
            if set(before["ranks"]) != expected_rank_keys or set(after["ranks"]) != expected_rank_keys:
                raise ValueError(f"{role}: incomplete snapshot inventory")
            cleanup_metrics(require(after["metrics"], f"{role} final server scrape"), side.data_parallel_size)
            counter_names = NIXL_COUNTERS if run.server.connector == "nixl" else MOONCAKE_COUNTERS
            server_counters[role] = engine_deltas(require(before["metrics"], f"{role} initial server scrape"),
                                                 require(after["metrics"], f"{role} final server scrape"), side.data_parallel_size, counter_names)
            for dp, tp in side.ranks(side.tp(run.server)):
                rank = f"dp{dp}.tp{tp}"
                left, right = before["ranks"][rank], after["ranks"][rank]
                old = require(left["steps"], f"{role}/{rank} startup log")
                full = require(right["steps"], f"{role}/{rank} final log")
                if not full.startswith(old) or (old and not old.endswith("\n")):
                    raise ValueError(f"{role}/{rank}: log reset or incomplete boundary")
                startup = [r for r in records(old) if any(k in r for k in ("kv_layout", "threads", "timing"))]
                required_startup = ("kv_layout", "timing") if dp == 0 else ("kv_layout",)
                if not all(any(k in r for r in startup) for k in required_startup):
                    raise ValueError(f"{role}/{rank}: missing startup layout/joint timing evidence")
                if dp == 0:
                    timing = [r["timing"] for r in startup if "timing" in r]
                    expected_args = dict(manifest["options"].get("timing_args") or {})
                    if receipt.get("timing"):
                        expected_args["path"] = receipt["timing"]["pod_path"]
                    if len(timing) != 1 or (timing[0].get("provider"), timing[0].get("args")) != (run.server.timing, expected_args):
                        raise ValueError(f"{role}/{rank}: timing provider/arguments differ from the resolved profile")
                if not any(r.get("threads") for r in records(full)):
                    raise ValueError(f"{role}/{rank}: missing native lazy thread evidence")
                layout = next(r["kv_layout"] for r in startup if "kv_layout" in r)
                if reference.uses_tent(run.server) and str(layout.get("tent_metrics_port")) != str(9100+dp*side.tp(run.server)+tp):
                    raise ValueError(f"{role}/{rank}: wrong TENT rank metrics port")
                if run.server.connector == "nixl" and (layout.get("nixl_transport") or {}).get("backends") != [reference.nixl_backend(run.server)]:
                    raise ValueError(f"{role}/{rank}: initialized backend identity missing or conflicting")
                segment = full[len(old):]
                rows = records(segment)
                times = [r[k] for r in rows for k in ("t", "t_start", "t_deadline", "t_scheduled") if k in r]
                if not times or any(not math.isfinite(t) or not before["started"] <= t <= after["finished"] for t in times):
                    raise ValueError(f"{role}/{rank}: log segment lies outside its local measurement boundaries")
                allocated = [request_uuid(req) for req in owned_ids(rows, run.server.enable_prefix_caching)]
                logs[(role, 0, dp, tp)] = "".join(json.dumps(r)+"\n" for r in startup) + segment
                # The union across DP engines must match clients, once per TP.
                for value in allocated:
                    if value not in expected_requests:
                        raise ValueError(f"{role}/{rank}: unowned request or unexpected warmup")
                if reference.uses_tent(run.server):
                    a = require(left["tent"], f"{role}/{rank} initial TENT scrape")
                    b = require(right["tent"], f"{role}/{rank} final TENT scrape")
                    parsed = series(b)
                    initial = series(a)
                    registered_pool(a, layout["bytes"])
                    registered_pool(b, layout["bytes"])
                    if any(name.endswith("_total") and key not in parsed for key in initial for name in (key[0],)):
                        raise ValueError(f"{role}/{rank}: TENT counter series disappeared")
                    counters = {}
                    names = {name for name, labels in parsed}
                    for name in names:
                        if name.endswith("_total"):
                            counters[name] = counter_delta(a, b, name, allow_new=True)
                        elif name == "tent_registered_buffer_bytes":
                            counters[name] = sum(v for (metric, _), v in parsed.items() if metric == name)
                    tent[role].append(counters)
                    backend_evidence[f"{role}/{rank}"] = {
                        "status": "scraped", "counter_epoch": before["server_processes"],
                        "delta": counters,
                        "failure_series": {name: "emitted" if name in names else "not emitted in either healthy scrape; native series is created on first event"
                                           for name in ("tent_task_failures_total", "tent_transport_attempt_failures_total")}}
                elif left["tent"].get("applicable") is not False or right["tent"].get("applicable") is not False:
                    raise ValueError("non-TENT backend received TENT evidence")
            a, b = require(before["network"], f"{role} network before"), require(after["network"], f"{role} network after")
            iface = manifest["interfaces"][role]
            if iface not in a or iface not in b:
                raise ValueError(f"{role}: missing interface {iface}")
            delta = [y-x for x, y in zip(a[iface], b[iface])]
            if len(delta) != 4 or any(n < 0 for n in delta):
                raise ValueError(f"{role}: interface counter reset")
            network[role] = {"interface": iface, "scope": "shared pod network namespace; all ranks once",
                             "rx_bytes": delta[0], "tx_bytes": delta[2], "local_interval": [before["started"], after["finished"]]}
        verdict = core(bench, logs, run, tent if reference.uses_tent(run.server) else None)
        problems.extend(verdict.pop("problems"))
        problems.extend(lifecycle_problems(logs, run, expected_requests))
        for role in ("prefill", "decode"):
            for tp in range(getattr(run, role).tp(run.server)):
                owners = defaultdict(list)
                for dp in range(getattr(run, role).data_parallel_size):
                    rows = records(logs[(role, 0, dp, tp)])
                    for req in owned_ids(rows, run.server.enable_prefix_caching):
                        owners[request_uuid(req)].append(dp)
                if set(owners) != expected_requests or any(len(v) != 1 for v in owners.values()):
                    problems.append(f"{role}/tp{tp}: client requests do not have exactly one DP owner")
        physical = verdict["physical_transfers"]["bytes"]
        if physical <= 0 or verdict["physical_transfers"]["count"] <= 0:
            problems.append("workload did not demonstrate any physical KV transfer")
        mover, other = ("decode", "prefill") if run.server.connector == "nixl" else ("prefill", "decode")
        native = server_counters[mover]
        descriptors = sum(operation["entry"][3] for operation in verdict["operations"])
        if (native[counter_names[0]], native[counter_names[1]], native[counter_names[2]]) != (
                physical, verdict["physical_transfers"]["count"], descriptors):
            problems.append("connector per-engine counters disagree with mover bytes/operations/descriptors")
        if any(server_counters[other][name] for name in counter_names[:3]):
            problems.append("non-mover unexpectedly reports physical payload/operations/descriptors")
        if network["prefill"]["tx_bytes"] < physical or network["decode"]["rx_bytes"] < physical:
            problems.append("interface traffic is less than connector payload")
        if manifest.get("profilers_stopped") is not True or manifest.get("samplers_stopped") is not True:
            problems.append("profilers/samplers not confirmed stopped")
        if (manifest.get("client_cleanup") or {}).get("stopped") is not True:
            problems.append("client process group not confirmed stopped")
        result.update(verdict, interface_bytes=network, client_ownership=mapped,
                      backend_evidence=backend_evidence if reference.uses_tent(run.server) else {"applicable": False},
                      server_counters=server_counters,
                      evidence_scope="Kubernetes raw evidence; no physical test result established offline")
    except (OSError, ValueError, KeyError, TypeError, IndexError, AssertionError) as error:
        problems.append(f"{type(error).__name__}: {error}")
    result["status"] = "passed" if not problems else "failed"
    return result


if __name__ == "__main__":
    import argparse
    command = argparse.ArgumentParser(description="Regenerate simulator accounting from a retained run directory")
    command.add_argument("directory", type=Path)
    args = command.parse_args()
    verdict = regenerate(args.directory)
    print(json.dumps(verdict, indent=2))
    raise SystemExit(0 if verdict["status"] == "passed" else 1)
