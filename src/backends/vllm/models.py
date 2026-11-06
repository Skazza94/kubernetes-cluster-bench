"""Pinned model configuration and tokenizer inputs; never upload weights."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from urllib.parse import quote
from urllib.request import Request, urlopen

HUB = "https://huggingface.co"
SKIPPED_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".msgpack", ".h5", ".onnx", ".ckpt")


def _get(url: str) -> bytes:
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        return response.read()


def files(directory: Path) -> list[Path]:
    # Select top-level configuration, tokenizer and model-code files.
    return sorted(p for p in directory.iterdir() if p.is_file()
                  and not p.name.endswith(SKIPPED_SUFFIXES) and not p.name.startswith("."))


def hashes(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files(directory)}


def fetch_model(repo: str, revision: str, cache: Path) -> Path:
    target = cache / repo.replace("/", "--") / revision
    receipt = target / ".files.json"
    if receipt.is_file():
        if hashes(target) != json.loads(receipt.read_text()):
            raise ValueError(f"cached model inputs changed: {target}")
        return target
    info = json.loads(_get(f"{HUB}/api/models/{repo}/revision/{revision}"))
    if info["sha"] != revision:
        raise ValueError("model hub resolved a different revision")
    names = sorted(e["rfilename"] for e in info.get("siblings", [])
                   if "/" not in e["rfilename"] and not e["rfilename"].startswith(".")
                   and not e["rfilename"].endswith(SKIPPED_SUFFIXES))
    if "config.json" not in names:
        raise ValueError("model revision contains no config.json")
    target.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    for name in names:
        content = _get(f"{HUB}/{repo}/resolve/{revision}/{quote(name)}")
        (target / name).write_bytes(content)
        downloaded[name] = hashlib.sha256(content).hexdigest()
    receipt.write_text(json.dumps(downloaded, indent=2) + "\n")
    return target


def model_inputs(repo: str, revision: str, local: str | None, cache: Path) -> tuple[bytes, dict]:
    """Capture one archive and derive its receipt from those exact bytes."""
    directory = Path(local).resolve() if local else fetch_model(repo, revision, cache)
    if not (directory / "config.json").is_file():
        raise ValueError("model inputs lack config.json")
    expected = json.loads((directory / ".files.json").read_text()) if local is None else None
    payload = tar_directory(directory)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        captured = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                    for m in archive if m.isfile()}
    if "config.json" not in captured:
        raise ValueError("captured model inputs lack config.json")
    if expected is not None and captured != expected:
        raise ValueError("captured model inputs differ from pinned downloaded files")
    # A local directory's claimed revision is not independently attested by its name.
    return payload, {"model": repo, "requested_revision": revision,
                     "revision_verified_by_hub": local is None,
                     "directory": str(directory), "sha256": captured,
                     "archive_sha256": hashlib.sha256(payload).hexdigest()}


def tar_directory(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in files(directory):
            content = path.read_bytes()  # Dereference once; tar never reopens the source.
            member = tarfile.TarInfo(path.name)
            member.size = len(content)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()
