# kubernetes-cluster-bench

`kubernetes-cluster-bench` is a profile-driven benchmarking and graphing framework for Kubernetes workloads. It provides one frontend for multiple benchmark backends while keeping execution, parsing, statistics, and rendering backend-specific where necessary.

Current backends:

- **iperf3**: TCP throughput between Kubernetes pods.
- **NIXLBench**: GPU/CPU buffer-transfer throughput using the NIXLBench image, over the UCX, LIBFABRIC or Mooncake backend.
- **vLLM**: prefill/decode serving with the KV-cache handoff over NIXL, using the vLLM CPU simulator in [vllm-sim/](./vllm-sim/) (template: [profiles/vllm-template.yaml](./profiles/vllm-template.yaml)). Each run checks that decode resumed from prefill's KV; `--y latency` plots request latency.

Each run records throughput and the CPU each side used for it (see "CPU accounting").

## How it works

```text
YAML profile
    ↓
benchmark frontend
    ↓
backend runner → Kubernetes pods → raw logs/results
    ↓
graphing frontend
    ↓
backend parser/analyzer → shared renderer → PNG graph
```

The raw artifacts are authoritative. The framework does not require a normalized results file. Each backend owns its native command construction and parsing, while the frontends provide common profile loading, lifecycle handling, dimension validation, filtering, aggregation, and graph rendering.

## Requirements

- Python environment with dependencies from [requirements.txt](./requirements.txt).
- `kubectl` configured for the target Kubernetes cluster.
- Nodes and container images suitable for the selected backend.
- For NIXL, a NIXLBench image containing `nixlbench`. GPU-enabled nodes are needed by default; set `execution.nixl.gpu: false` to run on CPU-only nodes with a CPU-only image (see `images/Dockerfile.nixl-cpu`).

The benchmark expects a Python venv, create and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Building images

The Dockerfiles are in [images/](./images/):

- [images/Dockerfile.iperf3](./images/Dockerfile.iperf3): iperf3 runtime image.
- [images/Dockerfile.nixl-gpu](./images/Dockerfile.nixl-gpu): GPU NIXLBench image (CUDA 13). NIXL and UCX come from the `nixl-cu13` wheel, on a minimal CUDA base image.
- [images/Dockerfile.nixl-cpu](./images/Dockerfile.nixl-cpu): CPU-only NIXLBench image built from source on Ubuntu (UCX, libfabric, Mooncake, NIXL with the UCX, LIBFABRIC and Mooncake plugins, `nixlbench`). Use it with `execution.nixl.gpu: false`. Upstream, the Mooncake plugin rejects transfers of more than 1024 descriptors (`kMaxRequestCount`), which is fewer than a KV cache transfer carries, so the image raises it to 65536 (build argument `MOONCAKE_MAX_REQUESTS`).

The vLLM CPU simulator image is vLLM's own CPU image, built from the pinned checkout with our patches. `vllm-sim/setup.sh` clones it into `vllm-sim/vllm` and applies `vllm-sim/patches/`; then:

```bash
docker build -f vllm-sim/vllm/docker/Dockerfile.cpu --target vllm-openai \
  --build-arg VLLM_CPU_X86=true --build-arg ENABLE_AMX_FP8=0 --build-arg max_jobs=4 \
  -t <registry>/kubernetes-cluster-bench:vllm-sim-cpu vllm-sim/vllm
```

The image includes pinned, unpatched NIXL (with its Python bindings) and Mooncake with TENT's `hp_tcp`, for `kv_transfer_mode: real`. In real mode the backend first runs one small READ between the two pods through the same backend and settings (`servers/transport-check-*.log`) and stops if any byte differs.

Build the iperf3 image from the repository root:

```bash
docker build \
  -f images/Dockerfile.iperf3 \
  -t kubernetes-cluster-bench:local \
  .
```

Build the GPU NIXL image for the cluster architecture. It copies nothing from the build context, so use the small `images/` directory as the context. `--build-arg CUDA_VERSION=...` and `--build-arg NIXL_VERSION=...` override the CUDA base image and the NIXL version:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f images/Dockerfile.nixl-gpu \
  -t <registry>/nixl-benchmark:gpu \
  --push \
  images/
```

Build the CPU-only NIXL image. It copies nothing from the build context, so use the small `images/` directory as the context:

```bash
docker build \
  -f images/Dockerfile.nixl-cpu \
  -t <registry>/nixl-benchmark:cpu \
  images/
```

It builds UCX 1.22.0 and NIXL v1.4.1 from source (`--build-arg UCX_VERSION=...` and `--build-arg NIXL_VERSION=...` override them), so the first build takes a while. It supports DRAM to DRAM transfers only; VRAM is not available.

Push each image to a registry reachable by every node and place the resulting image reference in the profile. The Python benchmark runner does not build or push images.

## Running benchmarks

The benchmark frontend intentionally accepts only two options:

```text
--profile FILE
--help
```

All benchmark settings belong in the profile. Start from a template and keep your own cluster-specific copy in `profiles/local/`, which is git-ignored:

```bash
cp profiles/iperf3-template.yaml profiles/local/my-iperf3.yaml   # or nixl-template.yaml
$EDITOR profiles/local/my-iperf3.yaml                            # fill in nodes, namespace, image
./run-benchmark.sh --profile profiles/local/my-iperf3.yaml
```

The templates ([iperf3](./profiles/iperf3-template.yaml), [nixl](./profiles/nixl-template.yaml)) are commented and are checked by the test suite, so they always load. The other files in `profiles/` (`pirate-*.yaml`) are fixed profiles for a specific cluster and are kept as reproducible examples only.

The runner creates a rendered manifest under the selected output directory, deploys backend resources, runs the configured dimensions, stores raw artifacts, and removes resources unless `keep_resources` is true. Deployment cleanup is attempted even when startup is interrupted.

## Profile format

A profile contains six conceptual parts:

- `backend`: registered backend name.
- `name`: dataset/output directory name.
- `nodes`: Kubernetes node placement.
- `dimensions`: values swept or reported as benchmark dimensions.
- `execution`: common lifecycle settings and backend execution settings.
- `test_id`: optional Kubernetes/resource identity; generated if omitted.

### iperf3 profile

```yaml
backend: iperf3
name: my-iperf3

test_id: optional-stable-id

nodes:
  initiator: <initiator-node>
  target: <target-node>

dimensions:
  threads: "1:4"

execution:
  namespace: <namespace>
  image: <registry>/kubernetes-cluster-bench:iperf3
  log_directory: logs
  runs: 3
  timeout_seconds: 120
  keep_resources: false
  iperf3:
    duration_seconds: 5
    warmup_seconds: 1
```

`threads: "1:4"` runs thread counts 1, 2, 3, and 4. A range is `start[:end[:step]]`, so `"1:8:2"` is 1, 3, 5 and 7. For exact values, write a comma-separated list (`"1,2,4,8"`) or a YAML list (`[1, 2, 4, 8]`), and the two forms can be mixed in a string (`"1,2:8:2"` is 1, 2, 4, 6 and 8); repeated values run once. iperf3 maps the current thread count to `iperf3 -P N`. `runs` controls repeated framework runs for each thread count. iperf3 uses duration plus warmup; transfer-size mode is not supported.

### NIXL profile

```yaml
backend: nixl
name: my-nixl

nodes:
  initiator: <initiator-node>
  target: <target-node>

dimensions:
  threads: "1:4"
  block_size: "1K:8M"
  batch_size: "1:8"

execution:
  namespace: <namespace>
  image: <registry>/nixl-benchmark:<tag>
  log_directory: logs
  runs: 3                      # repeated runs per thread count, as for every backend
  timeout_seconds: 120
  keep_resources: false
  nixl:
    iterations: 100            # timed iterations of each nixlbench execution (--num_iter)
    warmup_iterations: 10      # --warmup_iter
    run_timeout_seconds: 1800  # optional, default 1800
    gpu: true                  # optional, default true
    backend: UCX               # optional: UCX (default), LIBFABRIC or Mooncake, see "NIXL backends"
    parallelism: threads       # optional: threads (default) or processes, see "NIXL parallelism"
```

`gpu: false` removes the `nvidia.com/gpu` request from both pods and replaces `UCX_TLS=tcp,self,cuda_copy` with `tcp,self`, so the pods can run on nodes without GPUs. The data path is DRAM to DRAM either way. Use an image that starts `nixlbench` without a CUDA driver, such as [Dockerfile.nixl-cpu](./images/Dockerfile.nixl-cpu). The GPU image ([Dockerfile.nixl-gpu](./images/Dockerfile.nixl-gpu)) needs the NVIDIA driver at run time.

`run_timeout_seconds` bounds each NIXLBench execution (one per thread count). If it expires, both processes are killed, the run is recorded as failed, and the benchmark continues with the next thread count. Unlike `timeout_seconds`, which covers pod readiness and initiator startup, it should be sized to the full block/batch sweep.

NIXL currently hardcodes these execution settings:

```text
runtime: ASIO
initiator memory: DRAM
target memory: DRAM
ASIO port: 12345 (each nixlbench pair of a run gets the next port)
```

The runner derives the ASIO address from the initiator pod IP (or from its secondary interface when a `network` section is set). It maps:

```text
threads            → --num_threads
iterations         → --num_iter
warmup_iterations  → --warmup_iter
block_size   → --start_block_size / --max_block_size
batch_size   → --start_batch_size / --max_batch_size
```

`block_size` accepts `K`, `M`, and `G` suffixes using powers of 1024. `block_size` and `batch_size` ranges must increase by powers of two. The framework calculates `total_buffer_size` per thread using the current image’s observed validation requirement:

```text
max_block_size × max_batch_size × num_threads
```

assuming one initiator and one target device, then passes the calculated value to NIXLBench. NIXLBench may adjust iteration counts to distribute work evenly across threads.

`execution.runs` means the same for every backend: how many times each thread count is run (see "Repeated runs"). The NIXL iteration counts are NIXL-specific and live under `execution.nixl`. Profiles written before this split are rejected with a pointer to the new names: `execution.nixl.iterations` is required, so an old `runs: 512` cannot silently become 512 repeated runs, and `warmup_runs` and `repetitions` name their replacements.

### Secondary network (Multus)

By default the iperf3 client connects to the server pod's Kubernetes Service, and NIXL uses the initiator pod's `eth0` IP, so traffic uses the pods' primary interface (`eth0`). To benchmark another NIC, attach it with [Multus](https://github.com/k8snetworkplumbingwg/multus-cni) and add a top-level `network` section. The initiator and target are configured separately:

```yaml
network:
  initiator:
    attachment: nic-a          # NetworkAttachmentDefinition name, or <namespace>/<name>
    interface: net1            # optional, default net1
    ip: 10.10.0.1/24           # optional
  target:
    attachment: nic-b
    interface: net1
    ip: 10.10.0.2/24
```

The `NetworkAttachmentDefinition`s must already exist in the cluster (the framework does not create them). Before creating any pods it checks for each one, and fails with an explicit error if the cluster has no `NetworkAttachmentDefinition` resource (Multus is not installed) or if a named attachment does not exist. If `attachment` has no namespace, Multus looks for it in `execution.namespace`. Each pod is annotated with `k8s.v1.cni.cncf.io/networks` for its own side, and the framework then:

1. Reads each pod's `k8s.v1.cni.cncf.io/network-status` annotation to find the IP on that side's `interface`. If the pod has no such annotation or interface, the run fails with an error naming what was found (typically Multus is missing or the attachment is wrong).
2. Uses those addresses instead of `eth0`:
   - **iperf3** runs the client as `iperf3 -c <target address> -B <initiator address>`, so the connection starts and ends on the secondary interfaces instead of using the Service.
   - **NIXL** uses the initiator's secondary address as the ASIO rendezvous address (`--asio_address`), and sets `UCX_NET_DEVICES=<interface>` in each pod so UCX carries the data over that interface and not `eth0`.

`ip` is optional. Without it the address comes from the attachment's IPAM. With it, the address is requested through Multus (`ips` in the annotation, which needs a static-capable IPAM) and the run fails before any traffic if the pod ends up with a different address. It must include a prefix length. The resolved addresses are recorded under `network` in each `*-command.json`.

Backends that do not support `network` reject the section instead of ignoring it. NIXL runs over TCP here (`UCX_TLS` is unchanged), so this selects the NIC, not RDMA.

### NIXL backends: `UCX`, `LIBFABRIC` and `Mooncake`

`execution.nixl.backend` chooses the NIXL plugin that moves the data (`nixlbench --backend`). All three are run over TCP, DRAM to DRAM, and produce the same table, so their results can be plotted side by side.

| `backend` | Notes |
|---|---|
| `UCX` (default) | Nothing is added; `network` pins it with `UCX_NET_DEVICES`. |
| `LIBFABRIC` | Runs on libfabric's TCP provider (`FI_PROVIDER=tcp`). The plugin only completes its connection handshake in a progress thread, so `--enable_pt=true` is always passed and `args.enable_pt: false` is rejected. `network` pins it with `FI_TCP_IFACE`. |
| `Mooncake` | Forces Mooncake's TCP transport (`MC_FORCE_TCP=1`). `network` makes each side advertise its secondary address (`NIXL_MOONCAKE_IP_ADDR`). `args.enable_pt` and `args.progress_threads` are rejected because the plugin has no progress thread support. Its TCP transport does all of its I/O on `MC_NUM_TCP_IO_THREADS` threads (default 1), which is what limits a single nixlbench pair, so the runner sets it to the thread count (and to 1 in `processes` mode, where each process has one thread); a value in `env` overrides it. The variable exists from Mooncake v0.3.14, which the CPU image builds; older versions ignore it. The values used are recorded as `nixl_env_applied` in `*-command.json`. Mooncake's newer engine, TENT, is built into the CPU image and has an `hp_tcp` transport with its own worker threads and multi-rail routing; it is switched on with `env: {MC_USE_TENT: 1}` and configured with an inline JSON in `MC_TENT_CONF` (see "Environment placeholders" below). With TENT on, the runner does not set `MC_FORCE_TCP`, because that would select the old `tcp` transport. |

The image must contain the plugin: [images/Dockerfile.nixl-cpu](./images/Dockerfile.nixl-cpu) builds all three, while the GPU image only has UCX. A missing plugin makes `nixlbench` fail at start-up and the run is recorded as failed. `env` accepts the chosen backend's own variables besides `NIXL_` and `OMP_` (`UCX_` for UCX, `FI_` for LIBFABRIC, `MC_` for Mooncake), and the defaults above can be overridden there. The variable that pins the network cannot be set while a `network` section is present. The backend is recorded as `nixl_backend` in each `*-command.json`, and `args`, `env` and `parallelism` work as described below for every backend. UCX-specific tuning (for example the variables in the next sections) does not apply to the others.

### Repeated runs

`execution.runs` (default `1`) runs each thread count that many times, one after the other, so a graph point is backed by several samples. For NIXL each run repeats the whole block/batch sweep of that thread count. Each run is recorded on its own (`run1`, `run2`, ... in the file names and `*-command.json`), the run summary counts them separately, and a failed run does not stop the others. The run time grows in proportion, so a few runs of a narrow sweep is usually the better use of time than many of a wide one.

When a graph point has several samples, the plotted value is their mean and the error bars are their sample standard deviation (for NIXL the bars have zero height with one run, because a spread cannot be estimated from one value; iperf3 then shows the spread of its intervals). Two runs give a rough spread, and at least three are needed for it to mean much.

### NIXL parallelism: `threads` or `processes`

`dimensions.threads` is the number of parallel flows, like `iperf3 -P N`. `execution.nixl.parallelism` chooses how NIXL builds them:

| `parallelism` | What runs for `threads: N` |
|---|---|
| `threads` (default) | One nixlbench pair with `--num_threads N`, running the whole block/batch sweep in one go. |
| `processes` | N nixlbench pairs with one thread each, running together, one (block size, batch size) point at a time. |

With the default UCX transport over TCP, the threads of one nixlbench pair do not add up on the receiving side. The initiator drives one UCX worker per thread, but the target progresses all of its workers from a single thread (`nixlUcxEngine::progress()` loops over them), and that thread also performs the `recv()` copies. The pair is therefore limited by what one core can receive, however many threads the initiator uses. `enable_pt` and `progress_threads` do not change this: `enable_pt` uses one shared thread for all workers, and `progress_threads` only splits large batches on the sending side. With `processes`, each pair has its own ASIO port and its own target process, so the receive work spreads over N cores. Use `threads` to measure NIXL's own multi-threading and `processes` to see how far TCP gets with enough cores; the graph x-axis is `threads` in both cases and the mode is recorded in each `*-command.json`.

```yaml
dimensions:
  threads: "1:4"
execution:
  nixl:
    parallelism: processes
```

In `processes` mode:

- The pods need cores for N processes and memory for N sets of buffers (`resources` applies to the pod, not per process). Count the busy threads of each process, not just the processes: nixlbench's polling loop always occupies one core, and a backend that adds a progress thread (LIBFABRIC) occupies a second one per process on each side. A CPU limit that is too low throttles the pod, which shows as throughput that falls instead of rising once N passes what the limit allows.
- Each point is a separate nixlbench execution, so `run_timeout_seconds` bounds one point, `warmup_iterations` are spent again at every point, and the sweep takes longer than in `threads` mode. A failing point stops the run and the error names it.
- The pairs do not start their timed phases at exactly the same moment, so the processes' own averages cannot simply be added: a pair that ran partly alone reports a rate the link only gave it for that part, and the sum can exceed what the link carries. The runner therefore records when each initiator's result row appeared (`row_seen_s`) and the bytes each pair moved in its timed phase (`timed_bytes`, computed with nixlbench's iteration rules), which places every pair's timed phase on a common clock. The result of a point is the bytes all pairs moved divided by the span from the first start to the last end. It equals the sum of the rates when the pairs ran together, and it is never above N times the slowest pair's rate. Each observation also stores `overlap` (the share of that span in which all pairs were running), `window_s`, `skew` (fastest rate over slowest) and the per-process rates; a low `overlap` means the result is an average over a period in which fewer pairs were transferring. The timing has a resolution of a few tens of milliseconds, so use enough `iterations` for each point to last several seconds. Runs recorded without timing (older datasets) fall back to N times the slowest rate, marked `aggregate_method: slowest_rate`, which is only an upper bound.
- Logs are named `<test_id>-t<N>-run<n>-b<block>-n<batch>-p<k>-initiator.log` (and `-target.log`), and `*-command.json` lists them under `points`.
- `pipeline_depth` is checked against one thread per process.

### NIXL tuning: `args` and `env`

`execution.nixl.args` passes extra flags to `nixlbench`, and `execution.nixl.env` sets environment variables on both `nixlbench` processes (the same values on the initiator and the target, except for the placeholders in "Environment placeholders"; names must start with the backend's prefix, see "NIXL backends"). Both are optional, and the values are recorded in each `*-command.json` (`nixl_args`, `nixl_env`).

```yaml
execution:
  nixl:
    args:
      pipeline_depth: 16       # --pipeline_depth=16
    env:
      UCX_ZCOPY_THRESH: 1K
      UCX_TCP_RX_SEG_SIZE: 128K
```

Supported `args`: `pipeline_depth` and `large_blk_iter_ftr` (integers, at least 1), `progress_threads` (integer, at least 0), `enable_pt`, `prepared_xfer`, `recreate_xfer`, `reregister_mem`, `use_hugepages`, `enable_vmm` and `check_consistency` (true or false), and `op_type` (`READ` or `WRITE`). Flags the runner computes from the profile (block and batch sizes, threads, iterations, buffer size, runtime and backend) are rejected with a pointer to the profile field that sets them.

`pipeline_depth` has a limit that comes from how `nixlbench` splits its iterations. It runs `iterations / threads` iterations per thread, and for blocks above 1 MiB divides that by `large_blk_iter_ftr` (16) again. The same applies to `warmup_iterations`. If a thread runs fewer iterations than `pipeline_depth`, `nixlbench` caps the depth and then aborts with "descriptor count ... is not evenly divisible by pipeline depth". The profile is therefore rejected before any pod is created unless, for every thread count in the sweep and for blocks above 1 MiB (when `block_size` goes beyond 1M), both `iterations` and `warmup_iterations` are at least `pipeline_depth × threads × 16`. For depth 16 and 8 threads that is 2048.

`env` names must be `UCX_*`, `NIXL_*` or `OMP_*` variables in capitals. `UCX_NET_DEVICES` is rejected when a `network` section is set, because the interface names there already set it. List UCX variables with `ucx_info -cf` in the image (it is `/opt/ucx/bin/ucx_info` in the CPU image).

With the UCX backend, when `env` is set, UCX is asked to report unknown variables, and a run fails if either process logs `unused environment variable: ...`, so a misspelled name cannot silently leave a setting unapplied.

### Environment placeholders

In `execution.nixl.env` values, `{threads}` is replaced by the thread count of the run (always 1 in `processes` mode), `{threads*N}` by N times it (for a setting that should scale faster than the thread count, such as connections per peer) and `{address}` by the address of the side that runs the process (its secondary-network address, so it needs a `network` section and is rejected without one). This lets one value describe both sides and every point of the sweep, for example a JSON setting whose worker count follows `threads`:

```yaml
env:
  MC_USE_TENT: 1
  MC_TENT_CONF: '{"transports": {"tcp": {"enable": false}, "rdma": {"enable": false}, "shm": {"enable": false}, "gds": {"enable": false}, "ascend_direct": {"enable": false}, "hp_tcp": {"enable": true, "advertise_address": "{address}", "rail_addresses": ["{address}"], "worker_count": {threads}, "connections_per_peer": {threads}, "max_outstanding_bytes": 68719476736, "max_outstanding_tasks": 32768}}}'
```

`max_outstanding_bytes` and `max_outstanding_tasks` cap how much `hp_tcp` keeps in flight. With the defaults, sweeps with large blocks, large batches and several threads fail with `No more transports available after hp_tcp failed` and `NIXL_ERR_BACKEND`, so raise them when the in-flight data (roughly block × batch × `pipeline_depth` × threads) is large.

### Failure handling

A failed run is recorded and the benchmark continues with the next run. `execution.max_consecutive_failures` (default `5`, `0` disables) aborts the benchmark after that many failed runs in a row, so a broken cluster does not burn hours on runs that cannot succeed. Resources are still cleaned up, and remaining runs are reported as "not run". Any success resets the counter. At the end a summary line (`N/M runs succeeded, K failed, J not run`) is printed and the exit status is 1 if any run failed or the benchmark aborted.

### Pod resources

`execution.resources` sets CPU and memory for every benchmark container, mirroring Kubernetes `requests` and `limits`:

```yaml
execution:
  resources:
    requests: {cpu: 4, memory: 8Gi}
    limits:   {cpu: 4, memory: 8Gi}
```

Only `cpu` and `memory` are supported, and only the values you give replace the manifest's. The built-in manifests use these defaults:

| Backend | `requests` | `limits` |
|---|---|---|
| iperf3 | cpu 1, memory 1Gi | cpu 8, memory 16Gi |
| nixl | cpu 1, memory 8Gi | cpu 8, memory 32Gi |
| vllm | cpu 1, memory 2Gi | cpu 8, memory 16Gi |

The CPU limit is a CFS quota. If the work needs more cores than the limit (for example 8 threads with `limits.cpu: 1`), the pod is throttled and the result reflects the limit, not the network. Setting requests equal to limits makes the pods Guaranteed QoS, which gives more repeatable results and lets a node's static CPU manager assign dedicated cores if it is configured for that. A request above its limit (including the manifest's default limit) is rejected before any pod is created. The overrides are recorded as `resource_overrides` in each `*-command.json`.

### CPU accounting

Every run records how much CPU each side used, so results can be compared by cores and not only by throughput. While a run executes, a small shell loop in each benchmark container prints its cgroup's CPU counters (`cpu.stat`, cgroup v2, or `cpuacct.usage` on v1) every 0.1 s, and the framework stamps each line with its own clock on arrival, so the nodes' clocks do not need to agree. The samples are saved next to the run as `<run>-cpu-initiator.log` and `<run>-cpu-target.log` (seconds, usage in µs, throttled µs, CFS periods, throttled periods), and listed as `cpu_files` in `*-command.json`. The initiator is the iperf3 client or the NIXL initiator, and the target is the iperf3 server or the NIXL target.

The CPU of a result is averaged over the same span as its throughput:

- iperf3: the run's last `duration_seconds`, which end when the client exits (the warmup is excluded).
- NIXL `threads`: each table row's timed phase, which ends when the row appears in the log and lasted the bytes it moved (`timed_bytes`, recorded under `rows`) over its rate. Warmup and setup between rows are excluded.
- NIXL `processes`: the point's window from the first pair's start to the last pair's end.

Once a second the same loop also reads every thread's CPU time from `/proc/*/task/*/stat`, using only shell builtins so it starts no processes, into `<run>-threads-initiator.log` and `<run>-threads-target.log` (`thread_files` in `*-command.json`). That shows where the cores go, for example the benchmark's own polling loop against the transport's workers.

Each observation stores `cpu_cores_initiator`, `cpu_cores_target`, `cpu_threads_<side>` (the busiest threads over the window, each with `tid`, `pid`, `name` and `cores`; one-second resolution, so it needs windows of a few seconds), the share of CFS periods in which the container was throttled (`cpu_throttled_<side>`, present when the pod has a CPU limit) and `cpu_window_s`. The count includes everything in the container, not only the benchmark process, which in these images is just the benchmark and the sampler. Rows shorter than about a second have too few samples to be precise, so give each one enough NIXL `iterations` to last several seconds. Sampling is best effort: a pod that cannot be sampled leaves an empty file and the run goes on, and such runs, like datasets recorded before this feature, simply have no CPU values. Plot the values with `--y cpu` or `--y cpu-per-gb` (see "Graphing").

### Image pull policy

`execution.image_pull_policy` sets `imagePullPolicy` on every benchmark pod: `Always` (default), `IfNotPresent`, or `Never`. Use `IfNotPresent` or `Never` for images loaded directly onto the nodes (e.g. a local build that is not in a registry), where `Always` would fail the pull.

### Profile validation

The loader rejects unknown fields. Common fields are:

```text
backend, name, test_id
network.initiator.{attachment,interface,ip}, network.target.{attachment,interface,ip} (iperf3 and nixl)
nodes.initiator, nodes.target
dimensions.threads
execution.namespace, execution.image, execution.log_directory
execution.runs, execution.timeout_seconds, execution.keep_resources
execution.max_consecutive_failures, execution.image_pull_policy
execution.resources.requests.{cpu,memory}, execution.resources.limits.{cpu,memory}
```

Backend-exposed fields are:

| Backend | Execution fields | Dimensions |
|---|---|---|
| iperf3 | `duration_seconds`, `warmup_seconds` | `threads` |
| nixl | `iterations`, `warmup_iterations`, `run_timeout_seconds`, `gpu`, `backend`, `parallelism`, `args`, `env` | `threads`, `block_size`, `batch_size` |
| vllm | `prompt_lengths`, `output_tokens`, `block_size`, `kv_cache_memory_bytes`, `max_model_len`, `max_num_batched_tokens`, `max_num_seqs`, `dtype`, `timing`, `timing_args`, `kv_transfer_mode`, `kv_transfer_timing`, `env`, `startup_timeout_seconds`, `request_timeout_seconds` | `threads` (requests in flight) |

Values are validated before Kubernetes resources are created.

## Output layout

Benchmark output is stored as `logs/<name>/`, not under a backend subdirectory:

```text
logs/<name>/
├── manifest.yaml
├── raw/
│   ├── t1/
│   │   ├── <run>-command.json
│   │   ├── <run>-initiator.log or <run>-client.json
│   │   ├── <run>-target.log
│   │   └── <run>-cpu-initiator.log, <run>-cpu-target.log
│   └── t2/
└── graphs/
```

The raw command metadata records the backend, nodes, image, parameters, translated native command, status, and timestamps. NIXL logs contain the initiator measurement table and target coordination log. A failed run may have incomplete artifacts and will be reported by graphing rather than silently ignored.

## Graphing

The graph frontend requires:

```bash
./plot-benchmark.sh \
  --backend BACKEND \
  --name DATASET \
  --x DIMENSION
```

Example for iperf3:

```bash
./plot-benchmark.sh \
  --backend iperf3 \
  --name my-iperf3 \
  --x threads
```

Example for one NIXL configuration:

```bash
./plot-benchmark.sh \
  --backend nixl \
  --name my-nixl \
  --x threads \
  --filter block_size=4M \
  --filter batch_size=1
```

Compare batch sizes as separate series:

```bash
./plot-benchmark.sh \
  --backend nixl \
  --name my-nixl \
  --x threads \
  --filter block_size=4M \
  --series batch_size
```

Compare block sizes as separate series:

```bash
./plot-benchmark.sh \
  --backend nixl \
  --name my-nixl \
  --x threads \
  --filter batch_size=1 \
  --series block_size
```

Plot block size on a logarithmic axis with base 2:

```bash
./plot-benchmark.sh \
  --backend nixl \
  --name my-nixl \
  --x block_size \
  --filter batch_size=1 \
  --filter threads=1 \
  --x-log 2
```

Supported graph dimensions are declared by each backend. Current dimensions are `threads` for iperf3 and `threads`, `block_size`, and `batch_size` for NIXL.

`--y` chooses the metric:

| `--y` | Plots |
|---|---|
| `throughput` (default) | Gbit/s |
| `cpu` | Cores used by each side over the measured span (see "CPU accounting") |
| `cpu-per-gb` | CPU time each side spends per GB moved, in core-seconds: its cores over the throughput in GB/s. Lower is better, and it does not depend on the link speed: multiplied by a link's rate in GB/s it gives the cores needed to fill that link at the same efficiency |

The CPU metrics draw one line per side (`side=initiator`, `side=target`); `--filter side=target` keeps one. Repetitions give a mean and error bars as for throughput. For example, the cores needed to fill the link as the thread count grows:

```bash
./plot-benchmark.sh \
  --backend nixl \
  --name my-nixl \
  --x threads \
  --filter block_size=1M \
  --filter batch_size=64 \
  --y cpu
```

Filters accept numeric values and, for byte dimensions, `K`, `M`, and `G` suffixes:

```text
block_size=1K
block_size=4M
block_size=1G
```

If a dimension varies but is neither the X axis, a filter, nor a series, graphing rejects the request. This prevents unrelated configurations from being silently averaged. Constant and filtered dimensions are included in the graph title. Legends are displayed outside the plot and series are assigned distinct colors.

Default graph path:

```text
logs/<name>/graphs/<metric>-<x-dimension>[__<option>...].png
```

With no options the file is `throughput-<x-dimension>.png` (`cpu-...` or `cpu-per-gb-...` for the CPU metrics). Every option that changes what the graph shows adds a part, joined with `__`, so different plots of one dataset do not overwrite each other:

| Option | Part added |
|---|---|
| `--filter NAME=VALUE` | `NAME-VALUE`, sorted by name, so their order does not matter |
| `--series DIM` | `by-DIM` (several series are joined with `-`) |
| `--percentile P` | `pP` |
| `--x-log BASE` | `xlogBASE` |
| `--max-throughput V` | `maxV` |

For example `--x threads --filter block_size=4M --series batch_size` writes `throughput-threads__block_size-4M__by-batch_size.png`. The same options always give the same name, so re-plotting overwrites only its own earlier graph. Characters other than letters, digits, dots and underscores become `-`, and very long names are shortened with a short hash. `--title` does not affect the name.

Use `--output FILE` to choose another path. Use `--title TEXT` to override the generated title. `--max-throughput VALUE` adds a reference line in Gbit/s (throughput only), and `--percentile PERCENT` selects a percentile before aggregation where supported.

## Current backend limitations

- Only throughput and the CPU used for it are measured.
- No live cluster validation, image discovery, or NIXL version discovery is performed.
- NIXL runtime settings are intentionally fixed as described above.
- NIXL image behavior depends on the NIXLBench version inside the selected image.
- Live Kubernetes execution is not covered by the unit tests; tests use mocked Kubernetes/process boundaries and embedded parser fixtures.

## Repository map

```text
images/       Backend image Dockerfiles
manifests/    Backend Kubernetes manifest templates
profiles/     Profile templates (profiles/local/ is git-ignored for your own)
src/          Framework, backends, graphing, and tests
logs/         Benchmark datasets and generated graphs
```
