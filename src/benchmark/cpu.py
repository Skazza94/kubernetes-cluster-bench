from __future__ import annotations

import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import IO, Mapping, Sequence

from ..common.types import JsonValue
from .process import terminate_process


SAMPLE_INTERVAL_SECONDS = 0.1
# Per-thread CPU is read every this many samples (once a second by default).
THREAD_SAMPLE_EVERY = 10
# How long sampling goes on after a run, so the last samples cover its end.
LINGER_SECONDS = 0.3
# How far a window may overrun the samples and still be measured (clamped).
EDGE_TOLERANCE_SECONDS = 0.3
ROLES = ("initiator", "target")
# Threads listed per side in an observation, busiest first.
TOP_THREADS = 24

# Every interval it prints "cpu <usage_usec> <throttled_usec> <nr_periods>
# <nr_throttled>" for the container's cgroup (v2 or, failing that, v1). Every
# THREAD_SAMPLE_EVERY intervals it also prints "sweep", then one
# "thr <tid> <pid> <utime+stime ticks> <name>" line per thread in the
# container, after a first "hz <clock ticks per second>" line. It uses only
# shell builtins in the loop, so sampling does not start processes that would
# add to what it measures. It writes its pid to the file given as $3: ending
# the local kubectl does not end it in the pod, STOP_SCRIPT does.
SAMPLER_SCRIPT = """
interval=$1
every=$2
[ -n "$3" ] && echo $$ > "$3"
echo "hz $(getconf CLK_TCK 2>/dev/null || echo 100)"
if [ -r /sys/fs/cgroup/cgroup.controllers ]; then
  version=2
else
  for d in /sys/fs/cgroup/cpu,cpuacct /sys/fs/cgroup/cpuacct /sys/fs/cgroup/cpu; do
    [ -r "$d/cpuacct.usage" ] && acct=$d && break
  done
  [ -n "$acct" ] || exit 3
  stat=/sys/fs/cgroup/cpu/cpu.stat
  [ -r "$stat" ] || stat=$acct/cpu.stat
  version=1
fi
cgroup() {
  u=0; t=0; p=0; n=0
  if [ "$version" = 2 ]; then
    while read -r k v; do
      case $k in
        usage_usec) u=$v ;;
        throttled_usec) t=$v ;;
        nr_periods) p=$v ;;
        nr_throttled) n=$v ;;
      esac
    done < /sys/fs/cgroup/cpu.stat
  else
    read -r ns < "$acct/cpuacct.usage"
    u=$((ns / 1000))
    if [ -r "$stat" ]; then
      while read -r k v; do
        case $k in
          throttled_time) t=$((v / 1000)) ;;
          nr_periods) p=$v ;;
          nr_throttled) n=$v ;;
        esac
      done < "$stat"
    fi
  fi
  echo "cpu $u $t $p $n"
}
threads() {
  echo sweep
  for task in /proc/[0-9]*/task/[0-9]*; do
    { read -r line < "$task/stat"; } 2>/dev/null || continue
    name=${line#*(}
    name=${name%)*}
    set -f
    set -- ${line##*) }
    set +f
    [ $# -ge 13 ] || continue
    shift 11
    pid=${task#/proc/}
    echo "thr ${task##*/} ${pid%%/*} $(($1 + $2)) $name"
  done
}
i=0
while :; do
  cgroup
  [ $((i % every)) -eq 0 ] && threads
  i=$((i + 1))
  sleep "$interval"
done
"""


# Where samplers record their pids: <prefix><id>.pid.
PIDFILE_PREFIX = "/tmp/bench-cpu-sampler-"
# Stops every sampler whose pidfile starts with $1 (TERM, then KILL) and
# prints "stopped <pid>" or "running <pid>" for each. A zombie (exited, not yet
# reaped by its parent) counts as stopped.
STOP_SCRIPT = """
alive() {
  { read -r line < "/proc/$1/stat"; } 2>/dev/null || return 1
  line=${line##*) }
  [ "${line%% *}" != Z ]
}
for f in "$1"*.pid; do
  [ -f "$f" ] || continue
  read -r p < "$f"
  kill "$p" 2>/dev/null
  i=0
  while alive "$p" && [ $i -lt 30 ]; do sleep 0.1; i=$((i + 1)); done
  alive "$p" && kill -9 "$p" 2>/dev/null && sleep 0.1
  if alive "$p"; then
    echo "running $p"
  else
    echo "stopped $p"; rm -f "$f"
  fi
done
"""


def cleanup_problems(cleanup: Mapping[str, str]) -> list[str]:
    """What a cleanup round (``CpuSampler.cleanups``) failed to confirm: each
    role's report must be "stopped <pid>" lines, or nothing to stop."""

    return [f"{role}: {report}" for role, report in cleanup.items()
            if any(not line.startswith("stopped ") for line in report.splitlines())]


def sampler_command(
    interval: float = SAMPLE_INTERVAL_SECONDS,
    thread_every: int = THREAD_SAMPLE_EVERY,
    pidfile: str = "",
) -> list[str]:
    return ["sh", "-c", SAMPLER_SCRIPT, "cpu-sampler", f"{interval:g}", str(thread_every), pidfile]


def stop_command(prefix: str = PIDFILE_PREFIX) -> list[str]:
    return ["sh", "-c", STOP_SCRIPT, "stop-cpu-samplers", prefix]


class CpuSampler:
    """Records the CPU time used by the benchmark pods while a run executes.

    One sampler process per pod prints the container cgroup's counters, and
    each line is stamped on arrival with the local monotonic clock relative to
    ``origin``, the same clock the runners use for their own timings, so no
    clock synchronization between nodes is needed. The samples go to one file
    per role: ``<seconds> <usage_usec> <throttled_usec> <nr_periods>
    <nr_throttled>`` per line. With ``thread_files``, each thread's CPU time
    goes to a second file per role: an ``hz <ticks per second>`` header, then
    per sweep a ``<seconds> sweep`` line and ``<seconds> <tid> <pid> <ticks>
    <name>`` per thread.

    Sampling is best effort: if a pod cannot be sampled the file stays empty
    and the run goes on without CPU data. Each pod's samplers are stopped in
    the pod when sampling starts (one a failed run left behind) and ends;
    ``cleanups`` has, for each of those rounds, what the stop reported per
    role: "stopped <pid>" lines, "running <pid>" for one that survived, or
    "error: ..." when the stop itself failed (``cleanup_problems``).
    """

    def __init__(
        self,
        kubectl: object,
        namespace: str,
        pods: Mapping[str, str],
        files: Mapping[str, Path],
        origin: float,
        interval: float = SAMPLE_INTERVAL_SECONDS,
        thread_files: Mapping[str, Path] | None = None,
    ) -> None:
        self._kubectl = kubectl
        self._namespace = namespace
        self._pods = dict(pods)
        self._files = dict(files)
        self._thread_files = dict(thread_files or {})
        self._origin = origin
        self._interval = interval
        self._processes: list[subprocess.Popen[bytes]] = []
        self._threads: list[threading.Thread] = []
        self._pidfile = f"{PIDFILE_PREFIX}{uuid.uuid4().hex}.pid"
        self.cleanups: list[dict[str, str]] = []

    def start(self) -> None:
        self.cleanups.append({})
        for role, pod in self._pods.items():
            path = self._files[role]
            thread_path = self._thread_files.get(role)
            for target in (path, thread_path):
                if target is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("", encoding="utf-8")
            self._stop_in_pod(role, pod)
            try:
                process = self._kubectl.start_stream(  # type: ignore[attr-defined]
                    self._namespace, pod,
                    sampler_command(self._interval, pidfile=self._pidfile),
                )
            except Exception:
                continue
            stdout = getattr(process, "stdout", None)
            if not isinstance(process, subprocess.Popen) or stdout is None:
                continue
            self._processes.append(process)
            thread = threading.Thread(
                target=self._record, args=(stdout, path, thread_path), daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        if self._processes:
            time.sleep(LINGER_SECONDS)
        self.cleanups.append({})
        for role, pod in self._pods.items():
            self._stop_in_pod(role, pod)
        for process in self._processes:
            terminate_process(process, timeout_seconds=2.0)
        for thread in self._threads:
            thread.join(timeout=2.0)
        for process in self._processes:
            if process.stdout is not None:
                process.stdout.close()
        self._processes.clear()
        self._threads.clear()

    def _stop_in_pod(self, role: str, pod: str) -> None:
        run = getattr(self._kubectl, "run", None)
        if run is None:
            return
        try:
            completed = run(["exec", "-n", self._namespace, pod, "--", *stop_command()],
                            capture_output=True, timeout=30)
            self.cleanups[-1][role] = str(getattr(completed, "stdout", "") or "").strip()
        except Exception as error:  # the pod may be gone
            self.cleanups[-1][role] = f"error: {error}"

    def __enter__(self) -> CpuSampler:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _record(self, stdout: IO[bytes], path: Path, thread_path: Path | None) -> None:
        try:
            with path.open("a", encoding="utf-8") as output:
                threads = thread_path.open("a", encoding="utf-8") if thread_path else None
                try:
                    for raw in stdout:
                        seen = time.monotonic() - self._origin
                        fields = raw.decode("utf-8", errors="replace").split()
                        if not fields:
                            continue
                        kind = fields[0]
                        if kind == "cpu" and len(fields) == 5:
                            output.write(f"{seen:.4f} {' '.join(fields[1:])}\n")
                            output.flush()
                        elif threads is None:
                            continue
                        elif kind == "hz" and len(fields) == 2:
                            threads.write(f"hz {fields[1]}\n")
                        elif kind == "sweep":
                            threads.write(f"{seen:.4f} sweep\n")
                        elif kind == "thr" and len(fields) >= 5:
                            threads.write(f"{seen:.4f} {' '.join(fields[1:])}\n")
                finally:
                    if threads is not None:
                        threads.close()
        except (OSError, ValueError):
            return


Sample = tuple[float, float, float, float, float]


def read_samples(path: Path) -> list[Sample]:
    """Samples as (seconds, usage s, throttled s, periods, throttled periods)."""

    samples: list[Sample] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return samples
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            seen, usage, throttled, periods, throttled_periods = (float(field) for field in fields)
        except ValueError:
            continue
        samples.append((seen, usage / 1e6, throttled / 1e6, periods, throttled_periods))
    return samples


def _at(samples: Sequence[Sample], moment: float, column: int) -> float | None:
    """A counter's value at ``moment``, interpolated between the samples around it."""

    for before, after in zip(samples, samples[1:]):
        if before[0] <= moment <= after[0]:
            span = after[0] - before[0]
            if span <= 0:
                return before[column]
            share = (moment - before[0]) / span
            return before[column] + share * (after[column] - before[column])
    return None


def cpu_usage(samples: Sequence[Sample], start: float, end: float) -> dict[str, float] | None:
    """Cores used, on average, between ``start`` and ``end``, and the throttled share.

    A window that overruns the samples by at most ``EDGE_TOLERANCE_SECONDS``
    (a run that ended between the last sample and the sampler stopping) is
    clamped to them. None when the samples do not cover the window.
    """

    if end <= start or len(samples) < 2:
        return None
    first, last = samples[0][0], samples[-1][0]
    if first - EDGE_TOLERANCE_SECONDS <= start < first:
        start = first
    if last < end <= last + EDGE_TOLERANCE_SECONDS:
        end = last
    if end <= start:
        return None
    usage = [_at(samples, moment, 1) for moment in (start, end)]
    if usage[0] is None or usage[1] is None:
        return None
    result = {"cores": max(0.0, usage[1] - usage[0]) / (end - start)}
    periods = [_at(samples, moment, 3) for moment in (start, end)]
    throttled = [_at(samples, moment, 4) for moment in (start, end)]
    if None not in periods and None not in throttled:
        elapsed_periods = periods[1] - periods[0]  # type: ignore[operator]
        if elapsed_periods > 0:
            result["throttled"] = (throttled[1] - throttled[0]) / elapsed_periods  # type: ignore[operator]
    return result


Sweep = tuple[float, dict[int, tuple[int, int, str]]]


def read_thread_sweeps(path: Path) -> tuple[float, list[Sweep]]:
    """Clock ticks per second and the sweeps, each (seconds, {tid: (pid, ticks, name)})."""

    hz = 100.0
    sweeps: list[Sweep] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return hz, sweeps
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "hz":
            try:
                hz = float(fields[1]) or hz
            except ValueError:
                pass
            continue
        try:
            seen = float(fields[0])
        except (ValueError, IndexError):
            continue
        if len(fields) == 2 and fields[1] == "sweep":
            sweeps.append((seen, {}))
        elif len(fields) >= 5 and sweeps:
            try:
                tid, pid, ticks = int(fields[1]), int(fields[2]), int(fields[3])
            except ValueError:
                continue
            sweeps[-1][1][tid] = (pid, ticks, " ".join(fields[4:]))
    return hz, sweeps


def thread_usage(
    hz: float,
    sweeps: Sequence[Sweep],
    start: float,
    end: float,
) -> list[dict[str, JsonValue]] | None:
    """Each thread's cores between the first and last sweep inside the window.

    Thread sweeps are about a second apart, so the window is narrowed to the
    sweeps it contains (None when it contains fewer than two). A thread that
    started inside the window counts from zero; one that ended inside it is
    left out. Busiest first.
    """

    # Not past the end: the benchmark may exit right after it, taking its threads along.
    inside = [sweep for sweep in sweeps if start - EDGE_TOLERANCE_SECONDS <= sweep[0] <= end]
    if len(inside) < 2 or inside[-1][0] <= inside[0][0]:
        return None
    (first_seen, first), (last_seen, last) = inside[0], inside[-1]
    elapsed = last_seen - first_seen
    threads: list[dict[str, JsonValue]] = []
    for tid, (pid, ticks, name) in last.items():
        before = first.get(tid)
        used = ticks - (before[1] if before is not None and before[0] == pid else 0)
        cores = max(0, used) / hz / elapsed
        if cores >= 0.005:
            threads.append({"tid": tid, "pid": pid, "name": name, "cores": round(cores, 3)})
    threads.sort(key=lambda thread: -float(thread["cores"]))  # type: ignore[arg-type]
    return threads


def cpu_metadata(
    run_directory: Path,
    metadata: object,
    start: float,
    end: float,
) -> dict[str, JsonValue]:
    """Observation metadata with each side's CPU over a window, or {} without samples.

    ``metadata`` is a run's command metadata: its ``cpu_files`` (and, when
    present, ``thread_files``) map each role to a file in ``run_directory``.
    """

    if not isinstance(metadata, dict):
        return {}
    files = metadata.get("cpu_files")
    thread_files = metadata.get("thread_files")
    details: dict[str, JsonValue] = {}
    for role in ROLES:
        name = files.get(role) if isinstance(files, dict) else None
        if isinstance(name, str):
            usage = cpu_usage(read_samples(run_directory / name), start, end)
            if usage is not None:
                details[f"cpu_cores_{role}"] = usage["cores"]
                if "throttled" in usage:
                    details[f"cpu_throttled_{role}"] = usage["throttled"]
        name = thread_files.get(role) if isinstance(thread_files, dict) else None
        if isinstance(name, str):
            threads = thread_usage(*read_thread_sweeps(run_directory / name), start, end)
            if threads is not None:
                details[f"cpu_threads_{role}"] = threads[:TOP_THREADS]  # type: ignore[assignment]
    if any(key.startswith("cpu_cores_") for key in details):
        details["cpu_window_s"] = end - start
    return details
