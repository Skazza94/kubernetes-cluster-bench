#!/usr/bin/env python3
"""vLLM CPU simulator: pinned checkout and image build for the `vllm` backend.

  checkout   clone the pinned vLLM revision into vllm-sim/vllm and apply patches/
             (an existing checkout is never reset)
  export     write the checkout's commits on top of the pin to patches/
  build      build the checkout's upstream CPU image (the backend runs it simulated)

The checkout (vllm-sim/vllm, git-ignored) is pinned in source.env. Our vLLM changes
are commits on top of the pin in the checkout, tracked here as patches/ (run
`export` after committing in the checkout). The image labels record the HEAD it
was built from. Import the built image into both nodes' container runtimes, then run the
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
PATCHES = HERE / "patches"
LOCAL_BRANCH = "bench"
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

    A fresh checkout is cloned at the pin on LOCAL_BRANCH, and patches/ is
    applied on top with `git am`. An existing one is never reset: it must come
    from the pinned repository and contain the pinned revision, so local
    commits on top of it (our changes) are kept; a mismatch with patches/ is
    reported.
    """

    pin = source_pin()
    if not CHECKOUT.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--single-branch", "--branch",
                        pin["VLLM_BRANCH"], pin["VLLM_REPOSITORY"], str(CHECKOUT)], check=True)
        if git("rev-parse", "HEAD").stdout.strip() != pin["VLLM_REVISION"]:
            git("fetch", "--depth", "1", "origin", pin["VLLM_REVISION"])
            git("checkout", "--detach", pin["VLLM_REVISION"])
        git("switch", "-c", LOCAL_BRANCH)
        patches = sorted(PATCHES.glob("*.patch"))
        if patches:
            git("am", "--quiet", *map(str, patches))
            print(f"Applied {len(patches)} patch(es) from {PATCHES}", flush=True)
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
    commits = git("log", "--format=%s", "--reverse", f"{pin['VLLM_REVISION']}..HEAD").stdout.splitlines()
    patched = [_patch_subject(path) for path in sorted(PATCHES.glob("*.patch"))]
    if commits != patched:
        print(f"warning: the checkout's {len(commits)} commit(s) on top of the pin differ from "
              f"the {len(patched)} patch(es) in {PATCHES}; run `export` to record them",
              file=sys.stderr, flush=True)
    return head


def _patch_subject(path: Path) -> str:
    """The commit subject of a `git format-patch` file (may span lines)."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Subject: "):
            subject = [line[len("Subject: "):]]
            for follow in lines[index + 1:]:
                if not follow.startswith(" "):
                    break
                subject.append(follow[1:])
            text = " ".join(subject)
            return text.split("] ", 1)[1] if text.startswith("[PATCH") else text
    return ""


def export() -> None:
    """Replace patches/ with the checkout's commits on top of the pin."""

    pin = source_pin()
    checkout()
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise RuntimeError(f"{CHECKOUT} has uncommitted changes; commit them first")
    PATCHES.mkdir(exist_ok=True)
    for old in PATCHES.glob("*.patch"):
        old.unlink()
    # --zero-commit and no signature keep the files stable across re-exports.
    git("format-patch", "--no-signature", "--zero-commit", "--quiet",
        "-o", str(PATCHES), f"{pin['VLLM_REVISION']}..HEAD")
    print(f"Exported {len(list(PATCHES.glob('*.patch')))} patch(es) to {PATCHES}")


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
    commands.add_parser("checkout", help="clone the pin and apply patches/, or check the checkout")
    commands.add_parser("export", help="write the checkout's commits to patches/")
    builder = commands.add_parser("build", help="build the vLLM CPU image (not pushed)")
    builder.add_argument("image", nargs="?", default=DEFAULT_IMAGE)
    builder.add_argument("--jobs", type=int, default=int(os.environ.get("VLLM_BUILD_JOBS", 4)),
                         help="vLLM compile jobs (default 4 or $VLLM_BUILD_JOBS)")
    args = parser.parse_args()
    if getattr(args, "jobs", 1) < 1:
        parser.error("--jobs must be positive")
    if args.command == "checkout":
        checkout()
    elif args.command == "export":
        export()
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
