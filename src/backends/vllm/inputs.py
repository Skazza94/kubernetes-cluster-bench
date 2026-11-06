"""Stage model, dataset and timing inputs with integrity checks."""
from __future__ import annotations

import hashlib
import json
import io
import shutil
import tarfile
from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile

from ...benchmark.contracts import BenchmarkConfig
from ...benchmark.kubernetes import KubectlClient
from .config import FORK, ROOT, MOONCAKE_PROXY, NIXL_PROXY, options
from .datasets import dataset_file
from .models import model_inputs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def dataset_snapshot(path: Path):
    """Stream into a private, read-only capture; never upload the mutable source."""
    with tempfile.TemporaryDirectory(prefix="kcb-vllm-dataset-") as directory:
        snapshot = Path(directory) / "dataset"
        digest, size = hashlib.sha256(), 0
        with path.open("rb") as source, snapshot.open("xb") as target:
            while chunk := source.read(1 << 20):
                target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        snapshot.chmod(0o400)
        yield snapshot, digest.hexdigest(), size


def proxy_source(config: BenchmarkConfig) -> Path:
    return FORK / (NIXL_PROXY if options(config)["connector"] == "nixl" else MOONCAKE_PROXY)


def source_receipt(config: BenchmarkConfig, proxy: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(FORK), *args], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("simulator source must be clean before recording its revision")
    return {"fork_commit": git("rev-parse", "HEAD"), "proxy_sha256": sha256(proxy),
            "dockerfiles": {name: sha256(FORK / "docker" / name) for name in ("Dockerfile.cpu", "Dockerfile.aisim")}}


def timing_input(config: BenchmarkConfig) -> tuple[Path | None, str | None, dict | None]:
    o = options(config)
    if o["timing"] not in ("table", "replay"):
        return None, None, None
    source = Path(o["timing_args"]["path"]).resolve()
    digest = sha256(source)
    remote = f"/tmp/bench-inputs/timing-{digest}.jsonl"
    return source, remote, {"path": str(source), "pod_path": remote, "sha256": digest}


def upload_command(destination: str) -> list[str]:
    return ["sh", "-c", 'mkdir -p "$(dirname "$1")" && cat > "$1"', "sh", destination]


def stage(config: BenchmarkConfig, kubectl: KubectlClient, pods: dict[str, str],
          artifacts: Path) -> tuple[dict, dict]:
    o = options(config)
    cache = ROOT / "logs" / ".vllm-cache"
    data, model = model_inputs(o["model"], o["model_revision"], o["model_path"], cache / "models")
    names, receipt = {}, {"model": model, "pod_hashes": {}}
    artifacts.mkdir(parents=True, exist_ok=False)
    (artifacts / "model.tar").write_bytes(data)
    (artifacts / "model").mkdir()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).name != member.name:
                raise ValueError("captured model archive contains unsafe member")
            (artifacts / "model" / member.name).write_bytes(archive.extractfile(member).read())
    receipt["model"]["artifact"] = "model.tar"
    def retain():
        (artifacts / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    def verify(role, files):
        expected = {path: digest for path, digest in files.items()}
        result = kubectl.run(["exec", "-n", config.namespace, pods[role], "--",
                              "/opt/venv/bin/python", "/bench/evidence.py", "hashes", json.dumps(expected)],
                             capture_output=True, timeout=300)
        actual = json.loads(result.stdout)
        receipt["pod_hashes"].setdefault(role, {}).update(actual)
        retain()
        if actual != expected:
            raise ValueError(f"{role}: staged input hashes differ: {actual}")
    retain()
    for pod in pods.values():
        kubectl.exec_with_input(config.namespace, pod, ["tar", "-xf", "-", "-C", "/model"], data, timeout=300)
    for role in pods:
        verify(role, {"/model/" + name: digest for name, digest in model["sha256"].items()})
    dataset = dataset_file(o["dataset_name"], o["dataset_path"], cache / "datasets", ROOT)
    if dataset is not None:
        with dataset_snapshot(dataset) as (snapshot, digest, size):
            remote = f"/tmp/bench-inputs/dataset-{digest}-{dataset.name}"
            shutil.copyfile(snapshot, artifacts / "dataset")
            receipt["dataset"] = {"path": str(dataset), "pod_path": remote, "sha256": digest,
                                  "bytes": size, "artifact": "dataset"}
            retain()
            kubectl.exec_with_input(config.namespace, pods["target"], upload_command(remote), snapshot, timeout=900)
            if sha256(snapshot) != digest:
                raise ValueError("captured dataset changed during upload")
        names["dataset_path"] = remote
        verify("target", {remote: digest})
    else:
        receipt["dataset"] = {"name": o["dataset_name"], "path": o["dataset_path"],
                              "content_hash_available": False,
                              "reason": "client-resolved HF dataset" if o["dataset_name"] == "hf" else "generated workload"}
    source, remote, timing = timing_input(config)
    if source is not None:
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != timing["sha256"]:
            raise ValueError("timing input changed before staging")
        (artifacts / "timing.jsonl").write_bytes(content)
        timing["artifact"] = "timing.jsonl"
        receipt["timing"] = timing
        retain()
        for role, pod in pods.items():
            kubectl.exec_with_input(config.namespace, pod, upload_command(remote), content, timeout=300)
            verify(role, {remote: timing["sha256"]})
        names["timing_path"] = remote
        receipt["timing"] = timing
    retain()
    return names, receipt
