#!/usr/bin/env python3
"""vLLM CPU simulator: pinned checkout and image build for the `vllm` backend.

  checkout   clone the pinned vLLM revision into vllm-sim/vllm (never resets local work)
  build      build the checkout's upstream CPU image (the backend runs it simulated)

The checkout (vllm-sim/vllm, git-ignored) is pinned in source.env; our vLLM changes
are commits on top of the pin, and the image labels record the HEAD it was built
from. Import the built image into both nodes' container runtimes, then run the
two-server P/D benchmark with a `backend: vllm` profile (profiles/vllm-template.yaml).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys


HERE = Path(__file__).resolve().parent
CHECKOUT = HERE / "vllm"
LOCAL_BRANCH = "codex/cpu-simulator"
DEFAULT_IMAGE = "kubernetes-cluster-bench:vllm-sim-cpu"


def source_pin() -> dict[str, str]:
    """VLLM_REPOSITORY, VLLM_BRANCH and VLLM_REVISION from source.env."""

    pin = {}
    for line in (HERE / "source.env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            pin[key.strip()] = value.strip()
    return pin


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(CHECKOUT), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check,
    )


def checkout() -> str:
    """Make sure vllm-sim/vllm holds the pinned revision; return its HEAD.

    A fresh checkout is cloned at the pin on LOCAL_BRANCH. An existing one is
    never reset: it must come from the pinned repository and contain the
    pinned revision, so local commits on top of it (our changes) are kept.
    """

    pin = source_pin()
    if not CHECKOUT.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--single-branch", "--branch",
                        pin["VLLM_BRANCH"], pin["VLLM_REPOSITORY"], str(CHECKOUT)], check=True)
        if git("rev-parse", "HEAD").stdout.strip() != pin["VLLM_REVISION"]:
            git("fetch", "--depth", "1", "origin", pin["VLLM_REVISION"])
            git("checkout", "--detach", pin["VLLM_REVISION"])
        git("switch", "-c", LOCAL_BRANCH)
    if not (CHECKOUT / ".git").is_dir():
        raise RuntimeError(f"not a standalone Git checkout: {CHECKOUT}")
    if git("remote", "get-url", "origin").stdout.strip() != pin["VLLM_REPOSITORY"]:
        raise RuntimeError(f"{CHECKOUT} has a different origin; leaving it untouched")
    if git("merge-base", "--is-ancestor", pin["VLLM_REVISION"], "HEAD", check=False).returncode:
        raise RuntimeError(
            f"{CHECKOUT} does not contain the pinned revision {pin['VLLM_REVISION']}; "
            "leaving it untouched"
        )
    head = git("rev-parse", "HEAD").stdout.strip()
    print(f"vLLM checkout {CHECKOUT}: HEAD {head} (pin {pin['VLLM_REVISION']})", flush=True)
    return head


def build(args: argparse.Namespace) -> None:
    """Build the checkout's upstream CPU image, labelled with the revision it came from."""

    head = checkout()
    dirty = bool(git("status", "--porcelain").stdout.strip())
    subprocess.run([
        "docker", "build", "--progress=plain", "--platform", "linux/amd64",
        "--build-arg", f"max_jobs={args.jobs}", "--build-arg", "VLLM_CPU_X86=true",
        "--build-arg", "ENABLE_AMX_FP8=0", "--target", "vllm-openai",
        "--label", f"org.opencontainers.image.source={source_pin()['VLLM_REPOSITORY']}",
        "--label", f"org.opencontainers.image.revision={head}",
        "--label", f"ai.kubernetes-cluster-bench.source-dirty={str(dirty).lower()}",
        "-f", str(CHECKOUT / "docker" / "Dockerfile.cpu"), "-t", args.image, str(CHECKOUT),
    ], check=True)
    print(f"Built {args.image} from vLLM {head}{' with uncommitted changes' if dirty else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("checkout", help="clone or check the pinned vLLM checkout")
    builder = commands.add_parser("build", help="build the vLLM CPU image (not pushed)")
    builder.add_argument("image", nargs="?", default=DEFAULT_IMAGE)
    builder.add_argument("--jobs", type=int, default=int(os.environ.get("VLLM_BUILD_JOBS", 4)),
                         help="vLLM compile jobs (default 4 or $VLLM_BUILD_JOBS)")
    args = parser.parse_args()
    if getattr(args, "jobs", 1) < 1:
        parser.error("--jobs must be positive")
    if args.command == "checkout":
        checkout()
    else:
        build(args)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"vllm_sim: {error}")
