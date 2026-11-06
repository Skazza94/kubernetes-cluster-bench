"""Regenerate workload, transfer and CPU measurements from retained artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import accounting
from .parser import StepLog
from .profile import summarize_profile

GIB = 2**30
RUN_METRICS = {
    "useful-kv-throughput": "GiB/s",
    "physical-kv-throughput": "GiB/s",
    "cpu": "cores",
    "cpu-per-gib-useful": "core-s/GiB",
    "cpu-per-gib-physical": "core-s/GiB",
}


def distribution(values, label):
    if any(value is None for value in values):
        return {"available":False,"count":len(values),"timestamps":label,"reason":"native records contain unknown durations"}
    values = sorted(values)
    if not values:
        return {"available": False, "count": 0, "timestamps": label}
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"invalid latency: {label}")
    return {"available": True, "count": len(values), "unit": "s", "timestamps": label,
            **{f"p{p}": values[max(0, math.ceil(p/100*len(values))-1)] for p in (50,90,99)},
            "max": values[-1]}


def ratio(numerator, denominator):
    return numerator/denominator if numerator is not None and denominator is not None and denominator > 0 else None


def regenerate(directory: Path, verdict=None):
    """Read raw evidence; cached accounting/report summaries are never inputs."""
    verdict = accounting.regenerate(directory) if verdict is None else verdict
    manifest = json.loads((directory / "manifest.json").read_text())
    options = manifest["options"]
    problems = list(verdict["problems"])
    report = {"status": "incomplete", "problems": problems, "accounting_status": verdict["status"],
              "image_requested": manifest["image_requested"], "configuration": options,
              "units": {"GiB": GIB, "CPU": "one core-second = one CPU used for one second"},
              "scope": "CPU-resident KV and native host/network work; no GPU staging or PCIe-copy measurement",
              "cpu_overlap": "process totals include threads; node totals overlap process CPU and are reported separately",
              "servers": {}, "observations": []}
    def raw(name):
        if name not in manifest["artifact_hashes"]:
            raise ValueError(f"missing retained/hashed evidence: {name}")
        return (directory/name).read_text()
    try:
        client = json.loads(raw("client.json"))
        ledger = accounting.records(raw("client-ledger.jsonl"))
        started = {row["client_id"]:row for row in ledger if row.get("event")=="started"}
        ended = {row["client_id"]:row for row in ledger if row.get("event") in ("finished","exception")}
        completed = sum(row.get("success") is True for row in ended.values())
        failed = sum(row.get("event")=="exception" or row.get("success") is False for row in ended.values())
        unfinished = len(started.keys()-ended.keys())
        report["requests"] = {"attempted":len(started),"completed":completed,"failed":failed,"unfinished":unfinished}
        times = [row["t"] for row in started.values()]
        end_times = [row["t"] for row in ended.values()]
        execution = client.get("client",{}).get("execution_window")
        if times and end_times and not unfinished:
            client_window = [min(times),max(end_times)]
            source = "client request-hook monotonic timestamps, including failed requests"
        elif execution:
            client_window = execution
            source = "full client execution envelope; request completion evidence incomplete"
        else:
            raise ValueError("client workload window unavailable")
        if any(not math.isfinite(t) for t in client_window) or client_window[1]<=client_window[0]:
            raise ValueError("invalid client workload interval")
        duration = client_window[1]-client_window[0]
        report["workload"] = {"duration_s":duration,"client_local_interval":client_window,"source":source,
                              "serving_predictions_depend_on_timing_model":True}
        byte_counts = verdict.get("bytes",{})
        report["bytes"] = byte_counts
        report["physical_transfers"] = {**verdict.get("physical_transfers",{}),
                                        "payload_operations":verdict.get("payload_operations"),
                                        "zero_payload_operations":verdict.get("zero_payload_operations")}
        report["interface_bytes"] = verdict.get("interface_bytes")
        useful = byte_counts.get("useful_kv_bytes") if byte_counts.get("available") else None
        physical = byte_counts.get("physical_payload_bytes") if byte_counts.get("available") else None
        throughput = {"useful_kv_gib_s":ratio(useful,duration*GIB),"physical_payload_gib_s":ratio(physical,duration*GIB),
                      "duration_basis":"client workload wall time; affected by compute timing and request pacing"}
        report["throughput"] = throughput
        operations = verdict.get("operations",[])
        payload = [operation["entry"] for operation in operations if operation["entry"][0]>0]
        report["latency"] = {
            "client_e2e":distribution([r["latency"] for r in ended.values() if r.get("success") is True],
                                      "native benchmark client output.latency; compute timing model dependent"),
            "connector_transfer":distribution([entry[1] for entry in payload],
                "native NIXL xferDuration telemetry" if options["connector"]=="nixl" else "Mooncake synchronous batch_transfer_sync_write call duration (perf_counter)"),
            "connector_post_or_wait":distribution([entry[2] for entry in payload],
                "native NIXL postDuration telemetry" if options["connector"]=="nixl" else "Mooncake sender wait before posting"),
        }
        if payload and not report["latency"]["connector_transfer"]["available"]:
            problems.append("required connector transfer timing unavailable")
        waits = []
        for role in ("prefill","decode"):
            before = json.loads(raw(f"{role}-before.json"))
            after = json.loads(raw(f"{role}-after.json"))
            steps = StepLog()
            for rank, final in after["ranks"].items():
                startup, segment = accounting.rank_log_segment(before["ranks"][rank],final,f"{role}/{rank}")
                rank_steps = StepLog.parse("".join(json.dumps(row)+"\n" for row in startup)+segment)
                steps.steps.extend(rank_steps.steps)
                steps.allocations.extend(rank_steps.allocations)
                steps.threads.update(rank_steps.threads)
                steps.thread_times.update(rank_steps.thread_times)
                for request,events in rank_steps.events.items():
                    combined = steps.events.setdefault(request,{})
                    for kind,stamp in events.items():
                        earliest = kind in ("allocated","reads_posted") or role=="prefill" and kind=="finished"
                        combined[kind] = (min if earliest else max)(combined.get(kind,stamp),stamp)
                if role=="decode":
                    waits += [events["reads_done"]-events["reads_posted"] for events in rank_steps.events.values()
                              if "reads_done" in events and "reads_posted" in events]
            limits = execution_limits(before,after)
            if not options["profile_interval_seconds"]:
                report["servers"][role] = {"available":False,"reason":"profile_interval_seconds is zero",
                                          "execution_limits":limits}
                continue
            profile_text = raw(f"{role}-profile.jsonl")
            profile_rows = accounting.records(profile_text)
            samples = [row for row in profile_rows if "t" in row]
            if len(samples)<2:
                raise ValueError(f"{role}: fewer than two profiler samples")
            envelope = manifest["execution_envelopes"][role]
            start, end = envelope["start"],envelope["end"]
            if verdict["status"]=="passed":
                if role=="prefill":
                    start,end = client_window
                    window_source = "client request-hook interval on the same server clock"
                else:
                    allocations = [time for time,_,_ in steps.allocations]
                    finishes = [event["finished"] for event in steps.events.values() if "finished" in event]
                    if not allocations or not finishes:
                        raise ValueError("decode workload window lacks request lifecycle evidence")
                    start,end = min(allocations),max(finishes)
                    window_source = "local first allocation to final scheduler finish; includes gaps between requests"
            else:
                window_source = "full local execution envelope, including failed/incomplete requests"
            if start<envelope["start"] or end>envelope["end"] or start>=end:
                raise ValueError(f"{role}: workload interval outside local execution envelope")
            windows = {"idle":[samples[0]["t"],envelope["start"]],"setup":[envelope["start"],start],"workload":[start,end],
                       "drain":[end,after["finished"]],"post_client_execution":[envelope["end"],after["finished"]]}
            measurements = {}
            for name,window in windows.items():
                if window[1]<=window[0]:
                    measurements[name] = {"available":False,"reason":"no positive interval"}
                    if name in ("idle","workload"):
                        problems.append(f"{role}: {name} measurement window unavailable")
                    continue
                measurements[name] = summarize_profile(profile_text,steps,role=="prefill",window=window,
                                                       servers=before["server_processes"],boot_id=before["boot_id"])
                measurements[name]["available"] = True
            workload = measurements["workload"]
            cpu = workload.get("cpu",{}).get("server",{}).get("core_seconds")
            if not workload.get("complete") or cpu is None:
                problems.append(f"{role}: incomplete workload CPU coverage")
            iface = manifest["interfaces"][role]
            if not workload.get("network",{}).get(iface,{}).get("complete"):
                problems.append(f"{role}: incomplete workload interface coverage")
            idle = measurements["idle"]
            if not idle.get("complete") or idle.get("cpu",{}).get("server",{}).get("core_seconds") is None:
                problems.append(f"{role}: incomplete idle CPU coverage")
            costs = {"useful_kv":ratio(cpu,ratio(useful,GIB)),"physical_payload":ratio(cpu,ratio(physical,GIB))}
            report["servers"][role] = {"available":True,"clock":"node_monotonic","transfer_role":"sender" if role=="prefill" else "receiver",
                                       "execution_limits":limits,
                                       "workload_window_source":window_source,
                                       "windows":measurements,"server_core_seconds":cpu,"core_seconds_per_gib":costs,
                                       "cost_denominators":{"useful_kv":"canonical useful KV, one logical copy",
                                                            "physical_payload":"mover payload including replicas, transferred padding and transfer-only data"}}
        report["latency"]["scheduler_observed_read_wait"] = distribution(waits,"each rank: reads_posted.t to reads_done.t; includes polling/scheduler observation delay")
        if useful is None or physical is None or useful<=0 or physical<=0:
            problems.append("positive useful and physical byte denominators required")
        if problems:
            return report
        report["status"] = "complete" if options["profile_interval_seconds"] else "profiling_disabled"
        for metric,key in (("useful-kv-throughput","useful_kv_gib_s"),("physical-kv-throughput","physical_payload_gib_s")):
            report["observations"].append({"metric":metric,"value":throughput[key],"unit":"GiB/s"})
        if report["status"]=="complete":
            for role,server in report["servers"].items():
                workload = server["windows"]["workload"]
                report["observations"] += [
                    {"metric":"cpu","value":server["server_core_seconds"]/workload["window_s"],"unit":"cores","side":role},
                    *({"metric":f"cpu-per-gib-{name}","value":server["core_seconds_per_gib"][key],"unit":"core-s/GiB","side":role}
                      for name,key in (("useful","useful_kv"),("physical","physical_payload")))]
    except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
        problems.append(f"{type(error).__name__}: {error}")
    return report


def execution_limits(before, after):
    snapshots = {phase:{"local_interval":[snapshot["started"],snapshot["finished"]],
                       **snapshot.get("cpu_limits",{"ok":False,"error":"CPU limits were not captured"})}
                 for phase,snapshot in (("before",before),("after",after))}
    available = all(row["ok"] and all(row["value"].get(field) is not None
                                      for field in ("cpu.max","cpuset.cpus.effective"))
                    for row in snapshots.values())
    result = {"available":available,"clock":"node_monotonic","scope":"container CPU quota/cpuset and process affinities",
              "snapshots":snapshots,"stable":None}
    if available:
        result["stable"] = all(snapshots["before"]["value"][field]==snapshots["after"]["value"][field]
                               for field in ("cpu.max","cpuset.cpus.effective"))
        result["stability_scope"] = "before/after snapshots only; intermediate changes are not observed"
    else:
        result["reason"] = "CPU quota or effective cpuset unavailable; see retained snapshots"
    return result


def render_text(report):
    lines = [f"Measurement status: {report['status']}",f"Requests: {json.dumps(report.get('requests'))}",
             f"Workload: {json.dumps(report.get('workload'))}",f"Bytes: {json.dumps(report.get('bytes'))}",
             f"Throughput: {json.dumps(report.get('throughput'))}",f"Latency: {json.dumps(report.get('latency'))}"]
    for role,server in report["servers"].items():
        lines.append(f"{role}: server core-seconds={server.get('server_core_seconds')}; core-seconds/GiB={server.get('core_seconds_per_gib')}")
        lines.append(f"  Execution limits: {json.dumps(server.get('execution_limits'))}")
        for name,window in server.get("windows",{}).items():
            lines.append(f"  {name}: interval={window.get('local_interval')}; coverage={window.get('coverage_fraction')}; CPU={json.dumps(window.get('cpu'))}")
            lines.append(f"    Cgroup throttling: {json.dumps(window.get('throttling'))}")
    lines += [report["cpu_overlap"],report["scope"],*(f"Problem: {problem}" for problem in report["problems"])]
    return "\n".join(lines)+"\n"


if __name__=="__main__":
    import argparse
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory",type=Path)
    cli.add_argument("--output",type=Path,help="new output directory for report.json and report.txt")
    args = cli.parse_args()
    result = regenerate(args.directory)
    if args.output:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        (args.output/"report.txt").write_text(render_text(result))
    print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(0 if result["status"] in ("complete","profiling_disabled") else 1)
