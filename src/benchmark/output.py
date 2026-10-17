from __future__ import annotations

from pathlib import Path

from ..common.console import Colors
from .artifacts import RunOutcome
from .contracts import BenchmarkConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def print_configuration(
    config: BenchmarkConfig,
    colors: Colors,
) -> None:
    duration = config.parameters.get("duration_seconds", 10)
    print(f"\n{colors.cyan}Benchmark configuration{colors.reset}")
    print(f"  {'Test ID:':<20} {config.test_id}")
    print(f"  {'Kubernetes context:':<20} {config.kube_context}")
    print(f"  {'Client node:':<20} {config.initiator_node}")
    print(f"  {'Server node:':<20} {config.target_node}")
    print(f"  {'Namespace:':<20} {config.namespace}")
    print(f"  {'Logs directory:':<20} {config.log_directory}")
    print(f"  {'Image:':<20} {config.image}")
    print(f"  {'Backend:':<20} {config.backend}")
    if config.network is not None:
        for label, side in (("Client", config.network.initiator), ("Server", config.network.target)):
            requested = f", ip {side.ip}" if side.ip else ""
            print(f"  {label + ' network:':<20} {side.attachment} ({side.interface}{requested})")
    for kind, values in (
        ("requests", config.resources.requests),
        ("limits", config.resources.limits),
    ):
        if values:
            settings = ", ".join(f"{name}={value}" for name, value in values.items())
            print(f"  {'Pod ' + kind + ':':<20} {settings}")
    print(f"  {'Threads:':<20} {config.parameters.get('threads_spec', '1')}")
    print(f"  {'Runs per test:':<20} {config.runs}")
    if config.backend != "nixl":
        print(f"  {'Protocol:':<20} TCP")
        print(f"  {'Duration:':<20} {duration} seconds")
        print(f"  {'Warmup:':<20} {config.parameters.get('warmup_seconds', 5)} seconds")
    else:
        print(f"  {'Runtime:':<20} ASIO")
        print(f"  {'Backend:':<20} {config.parameters.get('backend', 'UCX')}")
        print(f"  {'Initiator memory:':<20} DRAM")
        print(f"  {'Target memory:':<20} DRAM")
        print(
            f"  {'Block size:':<20} "
            f"{_format_bytes(config.parameters['block_size_start'])}:"
            f"{_format_bytes(config.parameters['block_size_max'])}"
        )
        max_transfer = int(config.parameters['block_size_max']) * int(config.parameters['batch_size_max'])
        threads = 1 if config.parameters.get("parallelism") == "processes" else int(config.parameters['threads'])
        total_buffer = max_transfer * threads
        print(f"  {'Total buffer:':<20} {_format_bytes(total_buffer)}")
        print(
            f"  {'Batch size:':<20} "
            f"{config.parameters['batch_size_start']}:{config.parameters['batch_size_max']}"
        )
        print(f"  {'Iterations:':<20} {config.parameters['iterations']}")
        print(f"  {'Warmup iterations:':<20} {config.parameters['warmup_iterations']}")
        if config.parameters.get("parallelism", "threads") == "processes":
            print(
                f"  {'Parallelism:':<20} processes (each thread count runs that many "
                "single-thread nixlbench pairs, one sweep point at a time)"
            )
        for label, key in (("NIXL args:", "args"), ("NIXL env:", "env")):
            settings = config.parameters.get(key)
            if settings:
                text = ", ".join(f"{name}={value}" for name, value in settings.items())
                print(f"  {label:<20} {text}")
    print()


def print_run_dimensions(
    config: BenchmarkConfig,
    colors: Colors,
) -> None:
    if config.backend != "nixl":
        return
    parameters = config.parameters
    delimiter = f"{colors.cyan}{'=' * 72}{colors.reset}"
    print(f"\n{delimiter}")
    print(f"{colors.cyan}NIXL run dimensions{colors.reset}")
    processes = parameters.get("parallelism", "threads") == "processes"
    print(
        f"  {'Threads:':<20} {parameters.get('threads')}"
        + (" (processes of 1 thread each)" if processes else "")
    )
    print(
        f"  {'Block size:':<20} "
        f"{_format_bytes(parameters['block_size_start'])}:"
        f"{_format_bytes(parameters['block_size_max'])}"
    )
    max_transfer = int(parameters['block_size_max']) * int(parameters['batch_size_max'])
    threads = 1 if processes else int(parameters['threads'])
    total_buffer = max_transfer * threads
    print(f"  {'Total buffer:':<20} {total_buffer} bytes")
    print(
        f"  {'Batch size:':<20} "
        f"{parameters.get('batch_size_start')}:{parameters.get('batch_size_max')}"
    )
    print(f"  {'Iterations:':<20} {parameters.get('iterations')}")
    print(f"  {'Warmup iterations:':<20} {parameters.get('warmup_iterations')}")
    print(f"  {'Run:':<20} {parameters.get('run_number')} of {config.runs}")
    print(f"{delimiter}\n")


def _format_bytes(value: object) -> str:
    number = int(value)
    for suffix, multiplier in (("G", 1024**3), ("M", 1024**2), ("K", 1024)):
        if number % multiplier == 0:
            return f"{number // multiplier}{suffix}"
    return f"{number}B"


def print_results(
    config: BenchmarkConfig,
    outcome: RunOutcome,
    colors: Colors,
) -> None:
    run_directory = outcome.raw_directory
    server_path = run_directory.parent / f"{config.test_id}-server.json"
    print(f"\n{colors.cyan}Results{colors.reset}")
    print(
        f"  {colors.cyan}Client result:{colors.reset} "
        f"{colors.green}{display_path(_client_path(config, run_directory, outcome.run_number))}{colors.reset}"
    )
    print(
        f"  {colors.cyan}Server log:{colors.reset}    "
        f"{colors.green}{display_path(server_path)}{colors.reset}"
    )
    print(
        f"  {colors.cyan}Raw artifacts:{colors.reset} "
        f"{colors.green}{display_path(run_directory)}{colors.reset}"
    )


def _client_path(config: BenchmarkConfig, run_directory: Path, run_number: int) -> Path:
    thread = run_directory.name.removeprefix("t")
    return run_directory / f"{config.test_id}-t{thread}-run{run_number}-client.json"
