"""Resolve dataset files from explicit local paths or URLs."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def dataset_file(name: str, path: str | None, cache: Path, root: Path) -> Path | None:
    if name == "hf":
        return None  # Dataset ID is interpreted by the image's vllm bench client.
    if name == "sharegpt" and not path:
        raise ValueError("dataset_path is required for sharegpt: provide a local file or URL")
    if path is None:
        return None
    if urlparse(path).scheme in ("http", "https"):
        return download(path, cache)
    local = Path(path) if Path(path).is_absolute() else root / path
    if not local.is_file():
        raise ValueError(f"vLLM dataset_path not found: {local}")
    return local


def download(url: str, cache: Path) -> Path:
    parsed = urlparse(url)
    target = cache / hashlib.sha256(url.encode()).hexdigest()[:16] / (Path(parsed.path).name or "dataset")
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if os.environ.get("HF_TOKEN") and parsed.hostname == "huggingface.co":
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    partial = target.with_name(target.name + ".partial")
    with urlopen(Request(url, headers=headers), timeout=60) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.replace(target)
    return target
