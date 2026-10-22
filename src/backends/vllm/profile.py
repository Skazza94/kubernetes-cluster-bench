"""Where each server's CPU and network went during a run.

``pod_helper.py profile`` samples a pod's counters on the node's monotonic
clock, the clock of that server's step log, so each sample interval can be
split by what the server was doing (``state_segments``):

- ``compute``: a simulated step was running;
- ``waiting``: no step, but KV in flight: decode waiting on a READ, prefill
  holding blocks for decode to read;
- ``idle``: neither.

Threads of the engine process carry ``label``, from the step log's
``threads`` records: a Python thread's name, ``CONNECTOR_THREAD`` for native
threads the KV connector started (NIXL and its backends: the transport), or
None for other native ones (torch, gloo, ZMQ), which /proc alone cannot tell
apart by name. ``kv_connector`` adds up the connector's threads.

CPU used while ``waiting`` or ``idle`` is not simulated compute; it may be
useful work (the transport moving a READ) or overhead (a loop polling).
``busy_not_computing`` lists the threads busy then, with what tells the two
apart without settling it: voluntary context switches per CPU-second (a
thread that blocks often is event-driven; one that rarely blocks polls) and
how often the thread was found running. The node's CPU (``host``) includes
everything else on it, such as the network stack's softirqs, which a pod's
cgroup misses.
"""

from __future__ import annotations

import json
import math
import statistics
from typing import Sequence

from .parser import StepLog


STATES = ("compute", "waiting", "idle")
# The simulator's label for native threads started with the KV connector.
CONNECTOR_THREAD = "native (KV connector: NIXL and its backends)"
HOST_COLUMNS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
# Threads listed per server, busiest first.
TOP_THREADS = 16
# A thread busier than this while its server only waits or idles is listed in
# busy_not_computing, if the server spent at least BUSY_MIN_SECONDS in that
# state: CPU is apportioned per sample interval, too coarse for briefer states.
BUSY_CORES = 0.2
BUSY_MIN_SECONDS = 1.0
# A thread waking this often, or this busy, is active (for shared cores).
ACTIVE_WAKEUPS_PER_S = 100
ACTIVE_CORES = 0.005


def _union(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlap(intervals: Sequence[tuple[float, float]], start: float, end: float) -> float:
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)


def state_segments(steps: StepLog, producer: bool, end: float) -> dict[str, list[tuple[float, float]]]:
    """Each state's intervals on the server's clock (``idle`` is the rest).
    KV still in flight at ``end`` counts as in flight until then."""

    compute = _union(steps.steps)
    waiting = []
    for events in steps.events.values():
        if producer:
            start, stop = events.get("finished"), events.get("sends_done")
        else:
            start = events.get("reads_posted")
            stop = events.get("reads_done", events.get("reads_failed"))
        if start is not None:
            waiting.append((start, end if stop is None else stop))
    return {"compute": compute, "waiting": _union(waiting)}


def _shares(segments: dict[str, list[tuple[float, float]]], start: float, end: float) -> dict[str, float]:
    """Seconds of [start, end] in each state; compute first, then waiting."""

    span = end - start
    compute = _overlap(segments["compute"], start, end)
    both = sum(_overlap(segments["waiting"], max(start, a), min(end, b))
               for a, b in segments["compute"] if b > start and a < end)
    waiting = _overlap(segments["waiting"], start, end) - both
    return {"compute": compute, "waiting": waiting, "idle": max(0.0, span - compute - waiting)}


def summarize_profile(text: str, steps: StepLog, producer: bool) -> dict:
    """One server's CPU and network, in total and by state, over the profile."""

    header: dict = {}
    samples: list[dict] = []
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue  # kubectl or Python noise
        if "hz" in record:
            header = record
        elif "t" in record:
            samples.append(record)
    if not header or len(samples) < 2:
        raise ValueError("the profile has fewer than two samples")
    hz, me = float(header["hz"]), header["self"]
    first, last = samples[0]["t"], samples[-1]["t"]
    segments = state_segments(steps, producer, last)
    seconds = dict.fromkeys(STATES, 0.0)
    pod = dict.fromkeys(STATES, 0.0)
    threads: dict[int, dict] = {}
    known: dict[int, list] = {row[0]: row[1:8] for row in samples[0]["th"]}
    allowed: dict[int, str] = {row[0]: row[8] for row in samples[0]["th"] if len(row) > 8}
    waits: dict[int, dict[str, int]] = {}
    seen_running: dict[int, list[int]] = {}  # tid: [found running, observations]
    for before, after in zip(samples, samples[1:]):
        shares = _shares(segments, before["t"], after["t"])
        span = after["t"] - before["t"]
        for state in STATES:
            seconds[state] += shares[state]
        if before["cg"] is not None and after["cg"] is not None:
            used = (after["cg"] - before["cg"]) / 1e6
            for state in STATES:
                pod[state] += used * shares[state] / span
        for tid, pid, name, utime, stime, voluntary, involuntary, cpu, *extra in after["th"]:
            previous = known.get(tid)
            known[tid] = [pid, name, utime, stime, voluntary, involuntary, cpu]
            if extra:  # profiles from before allowed CPUs and wchan were recorded lack them
                allowed[tid] = extra[0]
                observed = seen_running.setdefault(tid, [0, 0])
                observed[1] += 1
                if extra[1] in ("", "0"):
                    observed[0] += 1
                else:
                    counts = waits.setdefault(tid, {})
                    counts[extra[1]] = counts.get(extra[1], 0) + 1
            if previous is None or previous[0] != pid:
                continue  # a new thread: its first interval is unknown
            thread = threads.setdefault(tid, {
                "tid": tid, "pid": pid, "name": name, "user": 0.0, "system": 0.0,
                "voluntary": 0, "involuntary": 0, "cpus": set(), **{f"{s}_s": 0.0 for s in STATES},
            })
            user, system = (utime - previous[2]) / hz, (stime - previous[3]) / hz
            thread["user"] += user
            thread["system"] += system
            thread["voluntary"] += voluntary - previous[4]
            thread["involuntary"] += involuntary - previous[5]
            thread["cpus"].add(cpu)
            for state in STATES:
                thread[f"{state}_s"] += (user + system) * shares[state] / span
    window = last - first
    summary: dict = {
        "window_s": round(window, 3),
        "state_s": {state: round(value, 3) for state, value in seconds.items()},
        "pod_cores": round(sum(pod.values()) / window, 3),
        "pod_cores_by_state": {state: round(pod[state] / seconds[state], 3)
                               for state in STATES if seconds[state] > 0},
        "profiler_cores": round(sum(t["user"] + t["system"] for t in threads.values()
                                    if t["pid"] == me) / window, 3),
        "host": _host(samples, hz, window),
        "network": _network(samples, window),
        "tcp": {name: samples[-1]["tcp"][name] - samples[0]["tcp"].get(name, 0)
                for name in samples[-1]["tcp"]},
    }
    for name, mtu in header.get("mtu", {}).items():
        if name in summary["network"]:
            summary["network"][name]["mtu"] = mtu
    rows = []
    for thread in threads.values():
        used = thread["user"] + thread["system"]
        if thread["pid"] == me or (used / window < ACTIVE_CORES
                                   and thread["voluntary"] / window < ACTIVE_WAKEUPS_PER_S):
            continue
        row = {
            "name": thread["name"], "pid": thread["pid"], "tid": thread["tid"],
            "cores": round(used / window, 3),
            "system_share": round(thread["system"] / used, 2) if used else 0.0,
            "voluntary_per_s": round(thread["voluntary"] / window, 1),
            "involuntary_per_s": round(thread["involuntary"] / window, 1),
            # Blocks per second of CPU used: low for a thread that polls.
            "voluntary_per_cpu_s": round(thread["voluntary"] / used) if used else None,
            "cpus": sorted(thread["cpus"]),
            "allowed": allowed.get(thread["tid"]),
            # The kernel functions it was most often found waiting in, and how
            # often (of the samples in which it had run) it was found running.
            "waits_in": [wait for wait, _ in sorted(waits.get(thread["tid"], {}).items(),
                                                    key=lambda item: -item[1])[:2]],
            "found_running": (round(seen_running[thread["tid"]][0] / seen_running[thread["tid"]][1], 2)
                              if seen_running.get(thread["tid"], [0, 0])[1] else None),
        }
        for state in STATES:
            if seconds[state] > 0:
                row[f"cores_{state}"] = round(thread[f"{state}_s"] / seconds[state], 3)
        rows.append(row)
    engine = {thread["pid"] for tid, thread in threads.items() if tid in steps.threads}
    for row in rows:
        if row["pid"] in engine:
            row["label"] = steps.threads.get(row["tid"])
    connector = [thread for tid, thread in threads.items()
                 if steps.threads.get(tid) == CONNECTOR_THREAD]
    if connector:
        used = sum(t["user"] + t["system"] for t in connector)
        summary["kv_connector"] = {
            "threads": len(connector),
            "cores": round(used / window, 4),
            "voluntary_per_s": round(sum(t["voluntary"] for t in connector) / window, 1),
            "involuntary_per_s": round(sum(t["involuntary"] for t in connector) / window, 1),
            **{f"cores_{state}": round(sum(t[f"{state}_s"] for t in connector) / seconds[state], 4)
               for state in STATES if seconds[state] > 0},
            "allowed": sorted({allowed.get(t["tid"]) or "" for t in connector}),
        }
    rows.sort(key=lambda row: -row["cores"])
    summary["threads"] = rows[:TOP_THREADS]
    summary["busy_not_computing"] = [
        {key: row.get(key) for key in ("name", "tid", "label", "cores_waiting", "cores_idle",
                                       "voluntary_per_cpu_s", "found_running", "waits_in")}
        for row in rows
        if any(row.get(f"cores_{state}", 0.0) >= BUSY_CORES and seconds[state] >= BUSY_MIN_SECONDS
               for state in ("waiting", "idle"))
    ]
    summary["shared_cores"] = _shared_cores(rows, window)
    return summary


def _shared_cores(rows: list[dict], window: float) -> dict[str, list[dict]]:
    """Single CPUs that more than one active thread is pinned to: those
    threads take turns on that core (involuntary switches)."""

    pinned: dict[str, list[dict]] = {}
    for row in rows:
        cpu = row.get("allowed")
        if cpu and cpu.isdigit() and (row["cores"] >= ACTIVE_CORES
                                      or row["voluntary_per_s"] >= ACTIVE_WAKEUPS_PER_S):
            pinned.setdefault(cpu, []).append({
                key: row[key] for key in ("name", "tid", "label", "cores", "voluntary_per_s",
                                          "involuntary_per_s", "waits_in") if key in row})
    return {cpu: threads for cpu, threads in pinned.items() if len(threads) > 1}


def _host(samples: list[dict], hz: float, window: float) -> dict[str, float]:
    """The node's cores in each /proc/stat column (all of it, not just the pod)."""

    first, last = samples[0]["host"], samples[-1]["host"]
    return {column: round((b - a) / hz / window, 3)
            for column, a, b in zip(HOST_COLUMNS, first, last) if column != "idle"}


def _network(samples: list[dict], window: float) -> dict[str, dict]:
    """Each interface's traffic: totals, mean packet sizes and, each way, the
    busiest sample interval's rate."""

    result = {}
    for name, last in samples[-1]["net"].items():
        first = samples[0]["net"].get(name)
        if first is None:
            continue
        rx_bytes, rx_packets, tx_bytes, tx_packets = (b - a for a, b in zip(first, last))
        if not (rx_bytes or tx_bytes):
            continue
        rates = [((after["net"][name][0] - before["net"][name][0]) / (after["t"] - before["t"]),
                  (after["net"][name][2] - before["net"][name][2]) / (after["t"] - before["t"]))
                 for before, after in zip(samples, samples[1:])
                 if name in before["net"] and name in after["net"] and after["t"] > before["t"]]
        result[name] = {
            "rx_bytes": rx_bytes, "tx_bytes": tx_bytes,
            "rx_packets": rx_packets, "tx_packets": tx_packets,
            "rx_mean_packet": round(rx_bytes / rx_packets) if rx_packets else None,
            "tx_mean_packet": round(tx_bytes / tx_packets) if tx_packets else None,
            "rx_gbps": round(rx_bytes * 8 / window / 1e9, 6),
            "tx_gbps": round(tx_bytes * 8 / window / 1e9, 6),
            "peak_rx_gbps": round(max(rx for rx, _ in rates) * 8 / 1e9, 6),
            "peak_tx_gbps": round(max(tx for _, tx in rates) * 8 / 1e9, 6),
        }
    return result


def summarize_reads(steps: StepLog) -> dict | None:
    """Decode's NIXL reads, one by one: sizes, durations, posting time and
    descriptors (the connector's own telemetry)."""

    if not steps.reads:
        return None
    sizes = [read[0] for read in steps.reads]
    durations = [read[1] for read in steps.reads]
    posts = [read[2] for read in steps.reads]

    def ms(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {"p50": round(statistics.median(ordered) * 1e3, 3),
                "p90": round(ordered[math.ceil(0.9 * len(ordered)) - 1] * 1e3, 3),  # nearest rank
                "max": round(ordered[-1] * 1e3, 3)}

    return {
        "count": len(steps.reads),
        "bytes": sum(sizes),
        "duration_ms": ms(durations),
        "post_ms": ms(posts),
        "descriptors_mean": round(statistics.fmean(read[3] for read in steps.reads), 1),
        "gbps_p50": round(statistics.median(s * 8 / d / 1e9 for s, d in zip(sizes, durations) if d > 0), 3),
    }


def _prometheus(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """Each sample of a Prometheus text page by (name, sorted labels)."""

    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        series, _, value = line.rpartition(" ")
        name, _, labels = series.partition("{")
        pairs = tuple(sorted(
            (key.strip(), raw.strip().strip('"'))
            for key, _, raw in (item.partition("=") for item in labels.rstrip("}").split(",") if item)
        ))
        try:
            values[(name, pairs)] = float(value)
        except ValueError:
            continue
    return values


def summarize_tent(before: str, after: str) -> dict:
    """What TENT itself saw during a run (its metrics' change over it): per
    transport, its counters, and each histogram's count, mean and the upper
    bounds of the buckets holding the median and the 90th percentile."""

    start, end = _prometheus(before), _prometheus(after)
    delta = {key: value - start.get(key, 0.0) for key, value in end.items() if key[0].startswith("tent_")}
    result: dict[str, dict] = {}
    buckets: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for (name, labels), value in delta.items():
        label = dict(labels)
        transport = label.pop("transport", "all")
        if name.endswith("_bucket"):
            bound = label.pop("le")
            buckets.setdefault((name[:-len("_bucket")], transport), []).append(
                (math.inf if bound == "+Inf" else float(bound), value))
        elif not name.endswith(("_sum", "_count")) and value:
            key = name if not label else f"{name}{{{','.join(f'{k}={v}' for k, v in sorted(label.items()))}}}"
            result.setdefault(transport, {})[key] = value
    for (name, transport), counts in buckets.items():
        total = delta.get((f"{name}_count", (("transport", transport),)), 0.0)
        if not total:
            continue
        counts.sort()
        histogram = {"count": int(total),
                     "mean": round(delta.get((f"{name}_sum", (("transport", transport),)), 0.0) / total, 1)}
        for quantile in (0.5, 0.9):
            bound = next((bound for bound, cumulative in counts if cumulative >= quantile * total), math.inf)
            histogram[f"p{int(quantile * 100)}_le"] = "+Inf" if math.isinf(bound) else bound
        result.setdefault(transport, {})[name] = histogram
    return result
