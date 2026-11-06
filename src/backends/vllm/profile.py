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
threads the KV connector started (the native transport), or
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
CONNECTOR_THREAD = "native (KV connector: its transfer engine)"
HOST_COLUMNS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
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


def process_role(row, profiler, servers, inventory):
    pid, birth, parent, command, *_ = row
    if pid == profiler or "/bench/pod_helper.py profile" in command or "cpu-sampler" in command:
        return "profiler"
    if "/bench/ledger.py client " in command or "/bench/pod_helper.py bench " in command:
        return "client"
    if "/bench/ledger.py proxy " in command:
        return "proxy"
    if (str(pid) in servers and servers[str(pid)]["start_ticks"] == birth
            or "vllm serve " in command or row[6].startswith("VLLM")):
        return "server"
    if parent in inventory and inventory[parent][1] <= birth and parent != pid:
        return process_role(inventory[parent],profiler,servers,inventory)
    return "other"


def summarize_profile(text: str, steps: StepLog, producer: bool, *,
                      window=None, servers=None, boot_id=None) -> dict:
    """Integrate the existing profiler on this server's local clock.

    Boundary intervals are apportioned uniformly. Gaps longer than three
    sampling periods are excluded. Sparse thread rows carry forward only
    while the thread is alive, and counters never bridge an identity change.
    Process totals include their threads; node totals overlap process CPU.
    """
    header, samples, known = {}, [], {}
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "hz" in row:
            if header:
                raise ValueError("profile has multiple headers")
            header = row
        elif "t" in row:
            if (len(row["host"]) != 8 or any(len(thread) != 11 for thread in row["th"])
                    or any(len(process) != 7 for process in row["ps"])):
                raise ValueError("incomplete profile counter/identity record")
            known.update({thread[0]: thread for thread in row["th"]})
            known = {tid: known[tid] for tid in row["alive"] if tid in known}
            if set(known) != set(row["alive"]):
                raise ValueError("profile thread inventory lacks counters")
            samples.append({**row, "observed": {thread[0] for thread in row["th"]}, "th": list(known.values())})
    if not header or len(samples) < 2:
        raise ValueError("the profile has fewer than two samples")
    if header.get("clock") != "node_monotonic":
        raise ValueError("profile must use this server's monotonic clock")
    if boot_id is not None and header.get("boot_id") != boot_id:
        raise ValueError("profiler boot identity differs from the server's counter epoch")
    hz, me, interval = float(header["hz"]), header["self"], float(header["interval"])
    if hz <= 0 or interval <= 0 or not math.isfinite(hz + interval):
        raise ValueError("invalid profiler clock or interval")
    if any(not math.isfinite(row["t"]) for row in samples) or any(
            b["t"] <= a["t"] for a, b in zip(samples, samples[1:])):
        raise ValueError("profile timestamps must increase")
    first, last = window if window is not None else (samples[0]["t"], samples[-1]["t"])
    if not math.isfinite(first + last) or last <= first:
        raise ValueError("invalid local profile window")
    duration = last - first
    segments = state_segments(steps, producer, last)
    seconds = _shares(segments, first, last)
    pod = dict.fromkeys(STATES, 0.0)
    threads = {}
    cpu = {role: {"observed_core_seconds": 0.0, "complete": True, "processes": set()}
           for role in ("server", "client", "proxy", "profiler", "other")}
    covered, gaps, network, host, tcp = 0.0, [], {}, dict.fromkeys(HOST_COLUMNS, 0.0), {}
    cg_covered = 0.0
    throttle = dict.fromkeys(("nr_periods","nr_throttled","throttled_usec"),0.0)
    throttle_covered = 0.0
    def delta(left, right):
        values = [b-a for a,b in zip(left,right)]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("profile counter reset or nonfinite counter")
        return values
    lifetimes = {}
    for row in samples:
        for thread in row["th"]:
            key = (thread[0],thread[1],thread[10])
            lifetimes.setdefault(key,[row["t"],row["t"]])[1] = row["t"]
    for before, after in zip(samples, samples[1:]):
        a, b = max(first, before["t"]), min(last, after["t"])
        if b <= a:
            continue
        span = after["t"] - before["t"]
        if span > 3 * interval:
            gaps.append([a,b])
            continue
        weight = (b-a)/span
        covered += b-a
        if all(isinstance(row.get("throttle"),dict) and all(key in row["throttle"] for key in throttle)
               for row in (before,after)):
            values = delta([before["throttle"][key] for key in throttle],
                           [after["throttle"][key] for key in throttle])
            for key,value in zip(throttle,values):
                throttle[key] += value*weight
            throttle_covered += b-a
        shares = _shares(segments,a,b)
        if before["cg"] is not None and after["cg"] is not None:
            used = delta([before["cg"]],[after["cg"]])[0] / 1e6 * weight
            cg_covered += b-a
            for state in STATES:
                pod[state] += used * shares[state] / (b-a)
        for column, used in zip(HOST_COLUMNS,delta(before["host"],after["host"])):
            host[column] += used/hz*weight
        for name in set(before["net"]) & set(after["net"]):
            values = delta(before["net"][name],after["net"][name])
            net = network.setdefault(name,{"totals":[0.0]*4,"coverage_s":0.0,"peak_rx_gbps":0.0,"peak_tx_gbps":0.0})
            net["totals"] = [old + value*weight for old,value in zip(net["totals"],values)]
            net["coverage_s"] += b-a
            net["peak_rx_gbps"] = max(net["peak_rx_gbps"],values[0]*8/span/1e9)
            net["peak_tx_gbps"] = max(net["peak_tx_gbps"],values[2]*8/span/1e9)
        for name in set(before["tcp"]) & set(after["tcp"]):
            tcp[name] = tcp.get(name,0) + delta([before["tcp"][name]],[after["tcp"][name]])[0]*weight
        left = {(row[0],row[1]):row for row in before["ps"]}
        right = {(row[0],row[1]):row for row in after["ps"]}
        inventory = {row[0]:row for row in (*left.values(),*right.values())}
        for identity in left.keys() | right.keys():
            process = right.get(identity,left.get(identity))
            role = process_role(process,me,servers or {},inventory)
            bucket = cpu[role]
            bucket["processes"].add(identity)
            if identity not in left or identity not in right:
                bucket["complete"] = False
                continue
            bucket["observed_core_seconds"] += sum(delta(left[identity][4:6],right[identity][4:6]))/hz*weight
        previous = {row[0]:row for row in before["th"]}
        for row in after["th"]:
            tid,pid,name,utime,stime,voluntary,involuntary,processor,allowed,wchan,birth = row
            old = previous.get(tid)
            key = (tid,pid,birth)
            if old is None or (old[1],old[10]) != (pid,birth):
                continue
            user,system,vol,invol = delta(old[3:7],row[3:7])
            thread = threads.setdefault(key,{"tid":tid,"pid":pid,"start_ticks":birth,"name":name,
                       "user":0.0,"system":0.0,"voluntary":0.0,"involuntary":0.0,"cpus":set(),
                       "allowed":allowed,"waits":{},"running":0,"observations":0,"coverage_s":0.0,
                       **{f"{state}_s":0.0 for state in STATES}})
            thread["process_role"] = process_role(inventory[pid],me,servers or {},inventory) if pid in inventory else "unknown"
            thread["user"] += user/hz*weight
            thread["system"] += system/hz*weight
            thread["voluntary"] += vol*weight
            thread["involuntary"] += invol*weight
            thread["cpus"].add(processor)
            thread["coverage_s"] += b-a
            if tid in after["observed"] and wchan is not None:
                thread["observations"] += 1
                thread["running"] += wchan in ("","0")
                if wchan not in ("","0"):
                    thread["waits"][wchan] = thread["waits"].get(wchan,0)+1
            for state in STATES:
                thread[f"{state}_s"] += (user+system)/hz * shares[state]/span
    complete = math.isclose(covered,duration,abs_tol=1e-8)
    for role,bucket in cpu.items():
        bucket["complete"] &= complete
        if role == "server" and not bucket["processes"]:
            bucket["complete"] = False
        if role == "server" and any((int(pid),value["start_ticks"]) not in bucket["processes"]
                                     for pid,value in (servers or {}).items()):
            bucket["complete"] = False
        bucket["processes"] = [{"pid":pid,"start_ticks":birth} for pid,birth in sorted(bucket["processes"])]
        bucket["core_seconds"] = bucket["observed_core_seconds"] if bucket["complete"] else None
        bucket["cores"] = bucket["core_seconds"]/duration if bucket["complete"] else None
    summary = {"window_s":duration,"local_interval":[first,last],"coverage_s":covered,
               "coverage_fraction":covered/duration,"complete":complete,"gaps":gaps,
               "boundary_interpolation":"uniform CPU/traffic within each adjacent sample interval",
               "state_s":{key:round(value,3) for key,value in seconds.items()},"cpu":cpu,
               "pod_cores":round(sum(pod.values())/duration,3) if math.isclose(cg_covered,duration,abs_tol=1e-8) else None,
               "pod_cores_by_state":{state:pod[state]/seconds[state] if math.isclose(cg_covered,duration,abs_tol=1e-8) else None
                                     for state in STATES if seconds[state]>0},
               "profiler_cores":cpu["profiler"]["cores"],
               "host":{name:round(value/duration,3) for name,value in host.items() if name!="idle"},
               "node_core_seconds":host if complete else None,
               "node_cpu_overlaps_process_cpu":True,"network":{},"tcp":tcp}
    throttle_complete = math.isclose(throttle_covered,duration,abs_tol=1e-8)
    summary["throttling"] = {"available":throttle_covered>0,"complete":throttle_complete,
        "clock":"node_monotonic","local_interval":[first,last],
        "scope":"container cgroup, including server, client, proxy and profilers",
        "coverage_s":throttle_covered,"coverage_fraction":throttle_covered/duration,
        "observed_delta":throttle if throttle_covered else None,
        "delta":throttle if throttle_complete else None,
        "throttled_seconds":throttle["throttled_usec"]/1e6 if throttle_complete else None,
        "throttled_period_fraction":throttle["nr_throttled"]/throttle["nr_periods"]
            if throttle_complete and throttle["nr_periods"] else None,
        "boundary_interpolation":"uniform counter deltas within adjacent local samples; counts may be fractional"}
    if not throttle_complete:
        summary["throttling"]["reason"] = "missing local cgroup v2 counters, sampling gaps or incomplete window coverage"
    for name,net in network.items():
        rx,rp,tx,tp = net.pop("totals")
        summary["network"][name] = {**net,"rx_bytes":rx,"rx_packets":rp,"tx_bytes":tx,"tx_packets":tp,
                                    "rx_mean_packet":round(rx/rp) if rp else None,"tx_mean_packet":round(tx/tp) if tp else None,
                                    "rx_gbps":rx*8/duration/1e9,"tx_gbps":tx*8/duration/1e9,
                                    "complete":math.isclose(net["coverage_s"],duration,abs_tol=1e-8),
                                    "mtu":header.get("mtu",{}).get(name)}
    rows,connector = [],[]
    for key,thread in threads.items():
        used = thread["user"]+thread["system"]
        label_time = steps.thread_times.get(thread["tid"])
        matched = label_time is not None and lifetimes[key][0] <= label_time <= lifetimes[key][1]
        label = steps.threads.get(thread["tid"]) if matched else None
        if label == CONNECTOR_THREAD:
            connector.append(thread)
        if thread["pid"]==me or (used/duration<ACTIVE_CORES and thread["voluntary"]/duration<ACTIVE_WAKEUPS_PER_S):
            continue
        row = {"tid":thread["tid"],"pid":thread["pid"],"start_ticks":thread["start_ticks"],"name":thread["name"],
               "process_role":thread["process_role"],"complete":complete and math.isclose(thread["coverage_s"],duration,abs_tol=1e-8),
               "label":label,"core_seconds":used,"coverage_s":thread["coverage_s"],"cores":round(used/duration,3),
               "system_share":round(thread["system"]/used,2) if used else 0.0,
               "voluntary_per_s":round(thread["voluntary"]/duration,1),"involuntary_per_s":round(thread["involuntary"]/duration,1),
               "voluntary_per_cpu_s":round(thread["voluntary"]/used) if used else None,
               "cpus":sorted(thread["cpus"]),"allowed":thread["allowed"],
               "waits_in":[name for name,_ in sorted(thread["waits"].items(),key=lambda item:-item[1])[:2]],
               "found_running":round(thread["running"]/thread["observations"],2) if thread["observations"] else None}
        row.update({f"cores_{state}":round(thread[f"{state}_s"]/seconds[state],3) for state in STATES if seconds[state]>0})
        rows.append(row)
    if connector:
        used = sum(t["user"]+t["system"] for t in connector)
        connector_complete = complete and all(math.isclose(t["coverage_s"],duration,abs_tol=1e-8) for t in connector)
        summary["kv_connector"] = {"available":True,"complete":connector_complete,"observed_core_seconds":used,
                  "core_seconds":used if connector_complete else None,
                  "threads":len(connector),"cores":round(used/duration,4) if connector_complete else None,
                  "voluntary_per_s":round(sum(t["voluntary"] for t in connector)/duration,1),
                  "involuntary_per_s":round(sum(t["involuntary"] for t in connector)/duration,1),
                  **{f"cores_{state}":round(sum(t[f"{state}_s"] for t in connector)/seconds[state],4) for state in STATES if seconds[state]>0},
                  "allowed":sorted({t["allowed"] for t in connector})}
    else:
        summary["kv_connector"] = {"available":False,"reason":"no connector label matched an observed thread lifetime"}
    rows.sort(key=lambda row:-row["cores"])
    summary["threads"] = rows
    summary["busy_not_computing"] = [{key:row.get(key) for key in ("name","tid","label","cores_waiting","cores_idle","voluntary_per_cpu_s","found_running","waits_in")}
         for row in rows if any(row.get(f"cores_{state}",0)>=BUSY_CORES and seconds[state]>=BUSY_MIN_SECONDS for state in ("waiting","idle"))]
    summary["shared_cores"] = _shared_cores(rows,duration)
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
