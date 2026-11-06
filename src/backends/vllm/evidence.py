"""Standard-library pod evidence capture. Failed reads retain an explicit error."""
from __future__ import annotations

import hashlib
import base64
from importlib import metadata
import json
from pathlib import Path
import signal
import subprocess
import sys
import sysconfig
import time
from urllib.request import urlopen


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def attempt(function):
    try:
        return {"ok": True, "value": function()}
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def processes(include_proxy=False):
    found = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            comm = (entry / "comm").read_text().strip()
            if not (comm.startswith("VLLM") or "vllm serve " in cmd or "vllm.entrypoints.openai" in cmd
                    or (include_proxy and "/bench/ledger.py proxy " in cmd)):
                continue
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] != "Z":
                found[entry.name] = {"start_ticks": int(stat[19]), "command": cmd, "comm": comm}
        except (OSError, ValueError, IndexError):
            continue
    return found


def rank_suffix(dp, tp):
    return (f".dp{dp}" if dp else "") + (f".tp{tp}" if tp else "")


def snapshot(dp_size: int, tp_size: int, tent: bool):
    # Read counters with the benchmark's profiler helpers.
    from pod_helper import interfaces, host_cpu
    start = time.monotonic()
    def fetch(port):
        with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as response:
            return response.read().decode()
    ranks = {}
    for dp in range(dp_size):
        for tp in range(tp_size):
            key = f"dp{dp}.tp{tp}"
            ranks[key] = {"steps": attempt(lambda: Path(f"/tmp/sim-steps{rank_suffix(dp, tp)}.jsonl").read_text()),
                          "tent": attempt(lambda: fetch(9100 + dp * tp_size + tp)) if tent else {
                              "applicable": False, "reason": "backend does not use TENT"}}
    versions = {}
    for name in ("vllm", "aisimulate", "nixl", "mooncake-transfer-engine", "torch"):
        versions[name] = attempt(lambda: metadata.version(name))
    for name, command in (("ucx", ["/opt/ucx/bin/ucx_info", "-v"]),
                          ("libfabric", ["/opt/libfabric/bin/fi_info", "--version"])):
        versions[name] = attempt(lambda command=command: subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10))
    # The image's Mooncake/TENT extension lacks wheel metadata.
    # Record its binary hashes.
    binaries = {}
    locations = [Path(sysconfig.get_paths()["purelib"]) / "mooncake"]
    locations += [Path("/opt") / package / "lib" for package in ("mooncake", "nixl", "ucx", "libfabric")]
    for location in locations:
        for path in sorted(location.rglob("*.so*")):
            if path.is_file() and not path.is_symlink():
                binaries[str(path)] = attempt(lambda path=path: file_hash(path))
    result = {"clock": "node_monotonic", "started": start,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "server_processes": processes(), "ranks": ranks,
            "metrics": attempt(lambda: fetch(8000)), "versions": versions, "native_binaries_sha256": binaries,
            "network": attempt(interfaces), "host_cpu": attempt(host_cpu)}
    result["finished"] = time.monotonic()
    return result


def stop_servers(timeout=5.0):
    owned = processes(include_proxy=True)
    for pid, original in owned.items():
        if processes(include_proxy=True).get(pid) == original:
            try:
                import os
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout
    while processes(include_proxy=True) and time.monotonic() < deadline:
        time.sleep(0.1)
    remaining = processes(include_proxy=True)
    # Report survivors; pod deletion remains the framework's owner-scoped action.
    return {"stopped": not remaining, "before": owned, "remaining": remaining}


def stop_client(directory, timeout=5.0):
    import os
    path = Path(directory) / "client-process.json"
    if not path.is_file():
        return {"stopped": False, "error": "missing client process receipt"}
    receipt = json.loads(path.read_text())
    pid = receipt["pid"]
    def alive():
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            return False
        if fields[0] == "Z":
            return False
        if receipt["start_ticks"] is None or int(fields[19]) != receipt["start_ticks"] or int(fields[2]) != receipt["process_group"]:
            raise ValueError("client PID/process group identity changed; refusing signal")
        return True
    if alive():
        os.killpg(receipt["process_group"], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while alive() and time.monotonic() < deadline:
        time.sleep(0.1)
    if alive():
        os.killpg(receipt["process_group"], signal.SIGKILL)
        time.sleep(0.1)
    remaining = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry/"stat").read_text().rsplit(")", 1)[1].split()
                if fields[0] != "Z" and int(fields[2]) == receipt["process_group"]:
                    remaining.append(int(entry.name))
            except (OSError, ValueError, IndexError):
                pass
    return {"stopped": not remaining, "remaining": remaining, "receipt": receipt}


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "hashes":
        value = {path: file_hash(path) for path in json.loads(args[0])}
    elif command == "snapshot":
        value = snapshot(int(args[0]), int(args[1]), args[2] == "true")
    elif command == "stop-servers":
        value = stop_servers()
    elif command == "stop-client":
        value = stop_client(args[0])
    elif command == "files":
        root = Path(args[0])
        value = {str(path.relative_to(root)): base64.b64encode(path.read_bytes()).decode()
                 for path in root.rglob("*") if path.is_file() and not path.is_symlink()}
    else:
        raise ValueError(command)
    print(json.dumps(value))
