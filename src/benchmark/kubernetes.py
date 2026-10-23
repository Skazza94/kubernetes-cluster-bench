from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Sequence


class KubectlError(RuntimeError):
    """Raised when kubectl cannot complete a command."""

    def __init__(self, arguments: Sequence[str], returncode: int, stderr: str) -> None:
        command = "kubectl " + " ".join(arguments)
        detail = stderr.strip() or f"exit status {returncode}"
        super().__init__(f"{command} failed: {detail}")
        self.arguments = tuple(arguments)
        self.returncode = returncode
        self.stderr = stderr


@dataclass(frozen=True)
class KubectlClient:
    """Small shared wrapper for backend Kubernetes operations."""

    executable: str = "kubectl"

    def run(
        self,
        arguments: Sequence[str],
        *,
        capture_output: bool = False,
        suppress_stdout: bool = False,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        stdout: int | None = subprocess.DEVNULL if suppress_stdout else None
        if capture_output:
            stdout = subprocess.PIPE
        result = subprocess.run(
            [self.executable, *arguments],
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr if isinstance(result.stderr, str) else ""
            raise KubectlError(arguments, result.returncode, stderr)
        return result

    def output(self, arguments: Sequence[str]) -> str:
        result = self.run(arguments, capture_output=True)
        return result.stdout.rstrip("\n")

    def current_context(self) -> str:
        try:
            return self.output(["config", "current-context"]) or "unknown"
        except (KubectlError, OSError):
            return "unknown"

    def resolve_namespace(self, requested: str | None = None) -> str:
        if requested:
            return requested
        try:
            namespace = self.output(
                [
                    "config",
                    "view",
                    "--minify",
                    "--output=jsonpath={.contexts[0].context.namespace}",
                ]
            )
        except (KubectlError, OSError):
            namespace = ""
        return namespace or "default"

    def context_and_namespace(self, requested: str | None = None) -> tuple[str, str]:
        return self.current_context(), self.resolve_namespace(requested)

    def resource_exists(self, resource: str, namespace: str) -> bool:
        result = subprocess.run(
            [
                self.executable,
                "get",
                "-n",
                namespace,
                resource,
                "--ignore-not-found",
                "-o",
                "name",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise KubectlError(
                result.args[1:],
                result.returncode,
                result.stderr,
            )
        return bool(result.stdout.strip())

    def apply_manifest(self, namespace: str, manifest: Path) -> None:
        self.run(
            ["apply", "-n", namespace, "-f", str(manifest)],
            suppress_stdout=True,
        )

    def wait_for(
        self,
        namespace: str,
        resource: str,
        condition: str,
        timeout_seconds: int,
    ) -> None:
        self.run(
            [
                "wait",
                "-n",
                namespace,
                f"--for={condition}",
                resource,
                f"--timeout={timeout_seconds}s",
            ],
            suppress_stdout=True,
        )

    def delete_manifest(self, namespace: str, manifest: Path) -> bool:
        result = subprocess.run(
            [
                self.executable,
                "delete",
                "-n",
                namespace,
                "-f",
                str(manifest),
                "--ignore-not-found",
            ],
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return result.returncode == 0

    def start_exec(
        self,
        namespace: str,
        pod: str,
        command: Sequence[str],
        stdout_path: Path,
    ) -> subprocess.Popen[bytes]:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        handle = stdout_path.open("wb")
        try:
            process = subprocess.Popen(
                [self.executable, "exec", "-n", namespace, pod, "--", *command],
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        except BaseException:
            handle.close()
            raise
        handle.close()
        return process

    def exec_with_input(
        self,
        namespace: str,
        pod: str,
        command: Sequence[str],
        data: bytes | Path,
        timeout: float | None = None,
    ) -> None:
        """Run a command in a pod with ``data`` (bytes, or a file's) on its
        standard input."""

        arguments = ["exec", "-i", "-n", namespace, pod, "--", *command]
        if isinstance(data, Path):
            with data.open("rb") as stdin:
                result = subprocess.run(
                    [self.executable, *arguments], stdin=stdin, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, check=False, timeout=timeout,
                )
        else:
            result = subprocess.run(
                [self.executable, *arguments], input=data, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, check=False, timeout=timeout,
            )
        if result.returncode != 0:
            raise KubectlError(arguments, result.returncode, result.stderr.decode(errors="replace"))

    def start_stream(
        self,
        namespace: str,
        pod: str,
        command: Sequence[str],
    ) -> subprocess.Popen[bytes]:
        """Start a command in a pod with its output readable line by line."""

        return subprocess.Popen(
            [self.executable, "exec", "-n", namespace, pod, "--", *command],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )


def wait_for_tcp_listener(
    client: KubectlClient,
    namespace: str,
    pod: str,
    port: int,
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> None:
    """Poll a pod's listening sockets for TCP-based backends."""

    import time

    deadline = time.monotonic() + timeout_seconds
    probe = ["exec", "-n", namespace, pod, "--", "ss", "-ltn"]
    while time.monotonic() < deadline:
        if process.poll() is not None:
            process.wait()
            raise RuntimeError("benchmark server exited during startup")
        remaining = max(0.1, deadline - time.monotonic())
        try:
            result = client.run(
                probe,
                capture_output=True,
                timeout=min(2.0, remaining),
            )
        except (KubectlError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and re.search(
            rf":{port}(?:\s|$)", result.stdout
        ):
            return
        time.sleep(0.2)
    raise TimeoutError("timed out waiting for benchmark server readiness")
