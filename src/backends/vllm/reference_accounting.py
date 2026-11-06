"""KV layout, rank ownership and transfer validation."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
DEFAULT_PROXY = Path("proxy.py")
NIXL_MAX_DESCRIPTORS = 1024
TENT_DEFAULT_MAX_OUTSTANDING_TASKS = 4096

@dataclass
class Side:
    node: str
    # A list: one per tensor-parallel rank, which every DP rank's TP rank of
    # that index uses (any prefill DP rank may pair with any decode DP rank).
    kv_network: str | list[str]
    # > 1: an MoE model's data-parallel attention with expert parallelism.
    data_parallel_size: int = 1
    tensor_parallel_size: int | None = None
    """This instance's TP size; None: the server's. Unequal sizes (one
    dividing the other) re-shard KV heads between ranks: both instances on
    one switched network (``kv_network`` its name), rank r on its node's
    link r."""

    def tp(self, server: "Server") -> int:
        return self.tensor_parallel_size or server.tensor_parallel_size

    def rank_networks(self, ranks: int) -> list[str]:
        if isinstance(self.kv_network, str):
            return [self.kv_network] * ranks
        return list(self.kv_network)

    def ranks(self, tp_size: int) -> list[tuple[int, int]]:
        """(DP rank, TP rank) of each of the instance's ranks, in the host's
        order (rank_addresses, TENT metrics ports, step logs)."""
        return [(dp, tp) for dp in range(self.data_parallel_size) for tp in range(tp_size)]



@dataclass
class Server:
    gpu: str = "h100"
    block_size: int | None = 16
    """Tokens per KV block; None: the platform's (a hybrid model's)."""
    enable_prefix_caching: bool | None = None
    """None: off with kv_check, else vLLM's default (on; a hybrid model's
    real transfers need it off). On with kv_check, a block's pattern is its
    content's, and prefix hits are checked too."""
    kv_cache_memory_bytes: int = 2 * 1024 ** 3
    max_model_len: int = 4096
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 32
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    """The KV cache dtype (--kv-cache-dtype, e.g. fp8); auto: the model's."""
    language_model_only: bool = False
    """A multimodal model served as its language model (--language-model-only):
    no modality, the encoders not built."""
    trust_remote_code: bool = False
    """Run the model repository's own code (--trust-remote-code, for the
    servers and the benchmark's tokenizer): Kimi's tokenizer is such code.
    Off unless a spec asks."""
    hp_tcp_args: dict = field(default_factory=dict)
    """TENT's hp_tcp transport settings (MC_TENT_CONF's transports.hp_tcp),
    e.g. {max_outstanding_tasks: 16384}. Admission, in the pinned TENT with
    hp_tcp alone and its runtime queue off: a submitted transfer's requests,
    after TENT merges those contiguous at both ends in the same
    registrations, are one task each, reserved together or not at all,
    without waiting, against the tasks still outstanding on that rank's
    transport (all its transfers in flight; each task frees its slot as it
    completes). Mooncake's Python binding retries a refused transfer once,
    at once. A transfer of more tasks than max_outstanding_tasks (4096 by
    default) is always refused; a smaller one may be, while others' tasks are
    outstanding. A connector's descriptors are not those tasks (TENT may
    merge them), and staying under the task limit leaves the byte limits
    (max_outstanding_bytes, max_transfer_bytes) and the transport's health
    to pass too. hp_tcp sends protocol headers and payloads separately. Small
    head-sliced tasks may be sensitive to Nagle/delayed-ACK interactions;
    task size relative to the effective TCP MSS, and therefore MTU, can
    affect transfer time."""
    env: dict = field(default_factory=dict)
    timing: str | None = None
    """Compute timing: zero, linear, table, replay or aisimulate."""
    timing_args: dict | None = None
    """Its arguments; aisimulate's default to {gpu} (plus a target_model)."""
    connector: str = "mooncake"
    """The KV connector: mooncake (MooncakeConnector, prefill writes) or nixl
    (NixlConnector, decode reads). nixl_backend selects Mooncake/TENT, UCX/TCP or LIBFABRIC/TCP."""
    nixl_backend: str = "mooncake"
    """NIXL backend: mooncake (default), ucx or libfabric over CPU TCP."""
    speculative: dict | None = None
    """Native MTP over NIXL: {method: mtp, num_speculative_tokens: k (1 to
    5), acceptance_rates: [...], seed: 0}. Both instances take the same
    speculative config (NIXL's handshake checks their transfer
    compatibility, the draft among it). The rates are the
    simulator's per-position conditional acceptance probabilities (the last
    repeats), sampled with the seed: a run input, not a model's."""



@dataclass
class Workload:
    input_len: int = 1024
    output_len: int = 16
    num_prompts: int = 40
    concurrency: int = 4
    seed: int = 0
    prefix_repetition: dict | None = None
    """Prompts sharing prefixes (the benchmark's prefix_repetition dataset):
    {prefix_len, suffix_len, num_prefixes}, input_len their sum. The text is
    tokenized again by the server: the sharing the checks report is the
    servers' own (their prefix hits), not this nominal split."""



@dataclass
class Run:
    name: str
    topology: str  # its name: the topology must be up
    model: Path  # a Hugging Face model directory (config and tokenizer)
    prefill: Side
    decode: Side
    client: str
    control_network: str
    kv_check: bool = True
    expect_fragmentation: bool = False
    """Require byte-checked transfers whose blocks are scattered on both
    instances (a tight KV budget's reuse); the counts are reported anyway."""
    server: Server = field(default_factory=Server)
    workload: Workload = field(default_factory=Workload)
    proxy: Path = DEFAULT_PROXY
    startup_timeout: float = 600.0



def uses_ucx(server: Server) -> bool:
    return server.connector == "nixl" and server.nixl_backend == "ucx"



def uses_tent(server: Server) -> bool:
    return server.connector != "nixl" or server.nixl_backend == "mooncake"



def nixl_backend(server: Server) -> str:
    return {"mooncake": "Mooncake", "ucx": "UCX", "libfabric": "LIBFABRIC"}[server.nixl_backend]



def _check_backend(server: Server) -> None:
    if server.nixl_backend not in ("mooncake", "ucx", "libfabric"):
        raise ValueError("server.nixl_backend is one of mooncake, ucx, libfabric")
    if server.nixl_backend != "mooncake" and server.connector != "nixl":
        raise ValueError("UCX/LIBFABRIC needs server.connector: nixl")
    if not uses_tent(server):
        if server.hp_tcp_args or any(k.startswith(("MC_", "TENT_", "NIXL_MOONCAKE_")) for k in server.env):
            raise ValueError("UCX/LIBFABRIC does not accept TENT/Mooncake settings")
    if uses_ucx(server):
        if "UCX_TLS" in server.env and server.env["UCX_TLS"] != "tcp,self":
            raise ValueError("Managed UCX needs UCX_TLS=tcp,self")
        if any(k in server.env for k in ("UCX_NET_DEVICES", "UCX_CONFIG_FILE")):
            raise ValueError("UCX interface/config selection is managed per rank")
        if any(k.startswith("NIXL_UCX_") and k != "NIXL_UCX_WARNING_TIMEOUT" for k in server.env):
            raise ValueError("Managed UCX does not accept NIXL_UCX overrides")
    if server.nixl_backend == "libfabric":
        policy = {"FI_PROVIDER": "tcp", "FI_ADDR_FORMAT": "FI_SOCKADDR_IN", "FI_PREFER_SYSCONFIG": "0"}
        diagnostics = {"FI_LOG_LEVEL", "FI_LOG_PROV", "FI_LOG_SUBSYS"}
        for key, value in server.env.items():
            if key.startswith("NIXL_LIBFABRIC_") or (key.startswith("FI_") and key not in set(policy) | diagnostics):
                raise ValueError(f"Managed LIBFABRIC does not accept {key}")
            if key in policy and str(value) != policy[key]:
                raise ValueError(f"Managed LIBFABRIC needs {key}={policy[key]}")



def descriptor_limit(server: Server) -> int | None:
    return NIXL_MAX_DESCRIPTORS if uses_tent(server) else None



def hp_tcp_task_limit(server: Server) -> int | None:
    """The hp_tcp task limit the servers ran with: hp_tcp_args', or a
    server.env MC_TENT_CONF's transport configuration; None if unknown
    (that config unreadable, or TENT not selected)."""

    def limit(hp_tcp: object) -> int | None:
        value = hp_tcp.get("max_outstanding_tasks", TENT_DEFAULT_MAX_OUTSTANDING_TASKS) if isinstance(
            hp_tcp, dict) else None
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    if not uses_tent(server):
        return None
    env = {name: str(value) for name, value in server.env.items()}
    if env.get("MC_USE_TENT", "1") != "1":
        return None
    if "MC_TENT_CONF" in env:
        try:
            return limit(json.loads(env["MC_TENT_CONF"])["transports"]["hp_tcp"])
        except (ValueError, KeyError, TypeError):
            return None
    return limit(server.hp_tcp_args)



def check(bench: dict, prefill_steps: list[str], decode_steps: list[str], routers: dict[str, int],
          run: Run, tent: dict[str, list[dict]] | None = None, ports: dict | None = None) -> dict:
    """The run's checks, from the client's results, every rank's step logs
    (``prefill_steps[i]``, ``decode_steps[i]``, in each side's `Side.ranks`
    order), the bytes each router received during the workload and each
    rank's TENT counters over it (``tent[role][i]``). Requests are counted
    once, from the client; bytes per rank, then summed. Each request is
    produced by one prefill DP rank and consumed by one decode DP rank,
    the same on every TP rank."""

    problems = []
    _check_backend(run.server)
    if not uses_tent(run.server) and tent:
        problems.append(f"{nixl_backend(run.server)} received unexpected TENT accounting")
    w = run.workload
    sizes = {"prefill": run.prefill.tp(run.server), "decode": run.decode.tp(run.server)}
    # Equal sizes pair rank r with rank r; unequal ones re-shard heads
    # (`_hetero_tp`).
    tp_size = sizes["prefill"] if sizes["prefill"] == sizes["decode"] else None
    # With prefix caching the checks also compare each request's prefix
    # hits (`kind` hit), on both instances, apart from its transfer.
    prefix = bool(run.server.enable_prefix_caching) and run.kv_check
    errors = [error for error in bench.get("errors", []) if error]
    if bench.get("completed") != w.num_prompts or errors:
        problems.append(f"{bench.get('completed')} of {w.num_prompts} request(s) completed; errors: {errors[:3]}")

    sides = {"prefill": run.prefill, "decode": run.decode}
    data_parallel = any(side.data_parallel_size > 1 for side in sides.values())

    def where(dp: int | None, tp: int) -> str:
        parts = ([f"DP rank {dp}"] if data_parallel and dp is not None else []) + (
            [f"rank {tp}"] if max(sizes.values()) > 1 else [])
        return " ".join(parts) + ": " if parts else ""

    logs = {role: dict(zip(side.ranks(sizes[role]), map(_records, steps)))
            for role, side, steps in (("prefill", run.prefill, prefill_steps),
                                      ("decode", run.decode, decode_steps))}
    tents = {role: dict(zip(sides[role].ranks(sizes[role]), tent[role])) for role in tent} if tent and uses_tent(run.server) else {}
    # Each TP rank's steps, every DP rank's requests: under MoE DP only DP
    # rank 0 logs the joint steps, the others their own KV.
    joint_steps: dict[tuple[str, int], dict[str, list[tuple[int, int, int]]]] = {
        (role, tp): {} for role in sides for tp in range(sizes[role])}
    for role in sides:
        for (dp, tp), records in logs[role].items():
            joint_steps[(role, tp)].update(_request_steps(records))
    ranks: dict[str, dict[tuple[int, int], dict]] = {role: {} for role in sides}
    # Mooncake producer ranks without a write series (unequal TP only).
    unwritten: set[int] = set()
    for role, side in sides.items():
        for (dp, tp), records in logs[role].items():
            at = where(dp, tp)
            layout = next((r["kv_layout"] for r in records if "kv_layout" in r), None)
            # Each log must be this rank's (not another's supplied twice).
            if layout is None:
                problems.append(f"{at}{role} logged no KV layout")
            elif (layout.get("tp_rank"), layout.get("tp_size")) != (tp, sizes[role]):
                problems.append(f"{at}{role}'s log is TP rank {layout.get('tp_rank')} of "
                                f"{layout.get('tp_size')}, not {tp} of {sizes[role]}")
            elif layout.get("dp_rank") != dp:
                problems.append(f"{at}{role}'s log is DP rank {layout.get('dp_rank')}, not {dp}")
            if layout is not None and not layout.get("groups"):
                raise ValueError(f"{at}{role}: grouped KV layout required")
            sent = {"count": 0, "bytes": 0, "failed": 0}
            for record in records:
                check = record.get("kv_check")
                if check and "error" not in check and not isinstance(check.get("groups"), list):
                    raise ValueError(f"{at}{role}: grouped KV check required")
                for transfers in (record.get("kv_transfers") or {}).values():
                    if any(len(entry) != 7 for entry in transfers.get("each", [])):
                        raise ValueError(f"{at}{role}: transfer records require seven fields")
                    for key in sent:
                        sent[key] += int(transfers.get(key) or 0)
            if sent["failed"]:
                problems.append(f"{at}{role}'s connector failed {sent['failed']} transfer(s)")
            identity = (layout or {}).get("nixl_transport")
            if run.server.connector == "nixl" and (not uses_tent(run.server) or identity is not None):
                expected = [nixl_backend(run.server)]
                if not isinstance(identity, dict) or identity.get("backends") != expected:
                    problems.append(f"{at}{role}'s initialized NIXL backend is {identity}, expected {expected}")
                elif not uses_tent(run.server):
                    binding = identity.get("binding") or {}
                    ucx = uses_ucx(run.server)
                    settings_ok = (binding.get("UCX_TLS") == "tcp,self"
                                   and binding.get("UCX_NET_DEVICES") == binding.get("interface")) if ucx else (
                        binding.get("FI_PROVIDER") == "tcp"
                        and binding.get("FI_TCP_IFACE") == binding.get("interface")
                        and binding.get("FI_ADDR_FORMAT") == "FI_SOCKADDR_IN"
                        and binding.get("FI_PREFER_SYSCONFIG") == "0"
                        and binding.get("config_file") == "/opt/libfabric/etc/libfabric.conf"
                        and binding.get("config_policy") == "absent_or_empty")
                    if not binding.get("address") or not binding.get("interface") or not settings_ok:
                        problems.append(f"{at}{role} has no valid {nixl_backend(run.server)} TCP/interface evidence: {binding}")
                    if (identity.get("progress_thread") is not True
                            or type(identity.get("num_threads")) is not int or identity["num_threads"] < 0):
                        problems.append(f"{at}{role} has no valid {nixl_backend(run.server)} thread/progress evidence")
                    if not ucx and (identity.get("num_threads") != 4
                                    or identity.get("thread_policy_source") != "agent_config"
                                    or identity.get("native_provider_observed") is not False
                                    or identity.get("native_threads_observed") is not False
                                    or identity.get("backend_options") != {"num_threads": "4"}
                                    or identity.get("plugin_defaults") != {"num_threads": "0", "split_batch_size": "1024"}):
                        problems.append(f"{at}{role} lacks LIBFABRIC configured-policy/backend-options evidence")
                    addresses = {p["address"] for network in (ports or {}).values() for p in network.values()
                                 if p.get("node") == side.node and p.get("slot") == tp and p.get("address")}
                    if addresses and addresses != {binding.get("address")}:
                        problems.append(f"{at}{role}'s {nixl_backend(run.server)} address {binding.get('address')} differs from rank port {addresses}")
            logged = [record["kv_check"] for record in records if "kv_check" in record]
            # A transfer's check; a prefix hit's; a request the content
            # patterns could not describe (fill, write).
            checks = [c for c in logged if c.get("kind") in (None, "transfer")]
            hits = [c for c in logged if c.get("kind") == "hit"]
            outside = [c for c in logged if c.get("kind") not in (None, "transfer", "hit")]
            if run.kv_check:
                bad = [c for c in checks if not c.get("ok") or c.get("mismatched_bytes")]
                if bad:
                    problems.append(f"{at}{len(bad)} request(s) received KV bytes unlike prefill's: {bad[0]}")
                bad = [c for c in hits if not c.get("ok") or c.get("mismatched_bytes")]
                if bad:
                    problems.append(f"{at}{role}: {len(bad)} prefix hit(s) held bytes unlike their content's: "
                                    f"{bad[0]}")
                if outside:
                    problems.append(f"{at}{role}: {len(outside)} request(s) outside the pattern lifecycle: "
                                    f"{outside[0]}")
            if run.kv_check and role == "decode":
                if layout:
                    # Failed receives are reported above; every other check is validated.
                    problems.extend(f"{at}{problem}" for problem in _byte_check_problems(
                        layout, [c for c in checks if "error" not in c], w.input_len, _pull_plans(records),
                        {req_id: steps[0][2] for req_id, steps in joint_steps[("decode", tp)].items()}, prefix))
            admissions = _admissions(records)
            # The hits each request's admissions imply, checked: only those
            # reconciled count (`valid_hits`).
            valid_hits = []
            if prefix and layout:
                found, valid_hits = _hit_problems(layout, hits, admissions)
                problems.extend(f"{at}{role}: {problem}" for problem in found)
            verified = {c["req_id"] for c in checks
                        if "req_id" in c and "error" not in c and c.get("ok") and not c.get("mismatched_bytes")}
            pulled = {request for request, adms in admissions.items() if any(
                a["pull"] and any(block for _, blocks in a["pull"].values() for block in blocks) for a in adms)}
            if prefix and role == "decode":
                # A request admitted without a pull is consumed through its
                # own hit, checked, of its whole prompt but the last token
                # (the one decode computes: native caching hits at most N - 1,
                # a hybrid's replay is that token too), as admitted and as its
                # first step computes; one pulled, through its byte check.
                hit_ids = {c["req_id"] for c in valid_hits}
                first_steps = joint_steps[("decode", tp)]
                for request in sorted(set(admissions) - pulled):
                    if request not in hit_ids:
                        problems.append(f"{at}decode admitted {request} with neither a pull nor a checked prefix hit")
                        continue
                    computed = admissions[request][0]["computed"]
                    first = (first_steps.get(request) or [None])[0]
                    if first is None:
                        problems.append(f"{at}decode admitted {request} without a pull and logged no step of it")
                    elif min(computed, first[0]) < first[2] - 1:
                        problems.append(
                            f"{at}decode admitted {request} without a pull, {computed} of its {first[2]} prompt "
                            f"tokens cached, its first step computing from {first[0]}: it prefilled "
                            f"{first[2] - min(computed, first[0])} tokens itself, not only the last (a transfer "
                            "replaced by local prefill)")
                if unverified := sorted(pulled - verified):
                    problems.append(f"{at}{len(unverified)} pulled request(s) without a successful byte check: "
                                    f"{unverified[0]}")
                posted = {request for r in records for request in (r.get("kv_events") or {}).get("reads_posted") or []}
                if unadmitted := sorted(posted - set(admissions)):
                    problems.append(f"{at}{len(unadmitted)} pull(s) posted for requests decode never admitted: "
                                    f"{unadmitted[0]}")
                if unposted := sorted({c["req_id"] for c in checks if "req_id" in c} - pulled):
                    problems.append(f"{at}{len(unposted)} byte check(s) of requests whose pull was not logged: "
                                    f"{unposted[0]}")
            # The requests the rank owns: prefill's allocations; decode's
            # admissions (prefix caching), its byte checks, else its
            # allocations.
            if prefix and role == "decode":
                owned = {_base_request_id(request) for request in admissions}
            elif role == "decode" and run.kv_check:
                owned = {_base_request_id(c["req_id"]) for c in checks if "req_id" in c}
            else:
                owned = {_base_request_id(req_id) for r in records
                         for req_id in ((r.get("kv_events") or {}).get("allocated") or {})}
            counters = tents.get(role, {}).get((dp, tp)) if tents else None
            if tents:
                # Mooncake's prefill writes the KV, NIXL's decode reads it.
                mover, moved_counter = ("decode", "tent_read_bytes_total") if (
                    run.server.connector == "nixl") else ("prefill", "tent_write_bytes_total")
                required = ("tent_registered_buffer_bytes",) + ((moved_counter,) if role == mover else ())
                if counters is None:
                    problems.append(f"{at}{role}'s TENT metrics did not answer")
                elif missing := [name for name in required if name not in counters]:
                    if tp_size is None and role == mover == "prefill" and missing == [moved_counter]:
                        # A replicated producer's rank that never writes never
                        # increments its write series: `_hetero_tp` accepts
                        # the absence only for a proven idle replica.
                        unwritten.add(tp)
                    else:
                        problems.append(f"{at}{role}'s TENT metrics lack {', '.join(missing)}")
                registered = (counters or {}).get("tent_registered_buffer_bytes")
                if registered is not None and layout and registered < layout["bytes"]:
                    problems.append(f"{at}{role}'s TENT registered {registered:.0f} bytes, less than its "
                                    f"{layout['bytes']} of KV")
                counted = (counters or {}).get(moved_counter)
                if role == mover and counted is not None and counted != sent["bytes"]:
                    problems.append(f"{at}TENT {'read' if mover == 'decode' else 'wrote'} {counted:.0f} bytes, "
                                    f"not the {sent['bytes']} its connector moved")
                if counters is not None:
                    # A healthy scrape omits a failure series never incremented.
                    failures = {key: int(counters.get(f"tent_{key}_total") or 0)
                                for key in ("task_failures", "transport_attempt_failures")}
                    if any(failures.values()):
                        problems.append(
                            f"{at}{role}'s TENT recorded {failures['task_failures']} failed task(s) and "
                            f"{failures['transport_attempt_failures']} failed transport attempt(s): not a clean "
                            f"transfer, whether or not a retry recovered it ({sent['failed']} connector transfer(s) "
                            "failed)")
            hp_tcp = None
            if tents and run.server.connector != "nixl" and role == "prefill":
                hp_tcp = _hp_tcp_writes(records, hp_tcp_task_limit(run.server))
            if tents and counters is not None:
                hp_tcp = {**(hp_tcp or {}), **{
                    key: int(counters.get(f"tent_{key}_total") or 0) if "failures" in key else (
                        None if counters.get(f"tent_{key}_total") is None else int(counters[f"tent_{key}_total"]))
                    for key in ("transport_attempts", "write_requests", "read_requests", "task_failures",
                                "transport_attempt_failures")}}
            ranks[role][(dp, tp)] = {
                "transfers": sent, "kv_checks": len(checks), "pulled": len(pulled),
                "checked_ids": {c["req_id"] for c in checks if "req_id" in c and "error" not in c},
                "verified_ids": verified,
                "allocated_ids": {req_id for r in records
                                  for req_id in ((r.get("kv_events") or {}).get("allocated") or {})},
                "hybrid": bool(layout and _prefill_backoff(layout)),
                "groups": layout and layout.get("groups") and _group_evidence(layout, checks, records),
                "kv_bytes_checked": sum(int(c.get("bytes") or 0) for c in checks),
                # Of them, NIXL's moves of regions holding none of a group's
                # layers (a shorter group's: a hybrid's state or attention).
                "kv_bytes_transfer_only": sum(int(g.get("transfer_only_bytes") or 0)
                                              for c in checks for g in c.get("groups") or []),
                "kv_bytes_spanned": _spanned_bytes(layout, checks) if layout else 0,
                # Prefix hits compared in place, reconciled with the
                # requests' admissions: no transfer moved them.
                "kv_hit_checks": len(valid_hits),
                "kv_hit_blocks": sum(c["blocks"] for c in valid_hits),
                "kv_bytes_hit_checked": sum(c["bytes"] for c in valid_hits),
                "requests": owned, "tent": counters, "hp_tcp": hp_tcp,
                "kv_layout": layout and {key: layout.get(key) for key in (
                    "dp_rank", "dp_local_rank", "tp_rank", "tp_size", "shape", "block_bytes", "bytes",
                    "tent_metrics_port")},
            }

    per_tp = []
    ownership: dict[str, dict[int, int]] = {"producers": {}, "consumers": {}}
    owners_by_tp: list[dict[str, dict[str, set[int]]]] = []
    hetero = None
    if tp_size is None:
        # Each decode rank, with the producer ranks the native pairing gives it.
        hetero = _hetero_tp(run, sizes, logs, ranks, ports, unwritten)
        problems.extend(hetero.pop("problems"))
    for tp in range(sizes["decode"] if hetero else tp_size):
        at = where(None, tp)
        first = hetero["sources"][tp][0] if hetero else tp
        of = {role: {dp: summary for (dp, rank), summary in ranks[role].items()
                     if rank == (first if role == "prefill" else tp)} for role in sides}
        checked = sum(summary["kv_bytes_checked"] for summary in of["decode"].values())
        moved = hetero["received"].get(tp, 0) if hetero else sum(
            summary["transfers"]["bytes"] for role in sides for summary in of[role].values())
        # Decode checks every block the connector pulls: the transfer plan.
        # What the checked blocks span as the connector moves them (a packed
        # block whole), which a packed layout's checked pages do not fill.
        spanned = sum(summary["kv_bytes_spanned"] for summary in of["decode"].values())
        if moved < spanned or (run.kv_check and moved != spanned):
            what = "decode checked" if spanned == checked else f"decode's checked blocks span ({checked} checked)"
            problems.append(f"{at}the connector moved {moved} bytes, not the {spanned} {what}")
        owners: dict[str, dict[str, set[int]]] = {}
        for role in sides:
            owners[role] = {}
            for dp, summary in of[role].items():
                for request in summary["requests"]:
                    owners[role].setdefault(request, set()).add(dp)
        # Requests the decode DP ranks own, and the byte checks they logged
        # (none without kv_check).
        consumed = len(set().union(*(summary["requests"] for summary in of["decode"].values())))
        byte_checks = sum(summary["kv_checks"] for summary in of["decode"].values())
        # With prefix caching a request served from decode's own hits pulls
        # nothing: one byte check for each request that pulled.
        pulls = sum(summary["pulled"] for summary in of["decode"].values()) if prefix else consumed
        if run.kv_check:
            if consumed != w.num_prompts:
                problems.append(f"{at}decode {'admitted' if prefix else 'byte-checked'} {consumed} distinct "
                                f"request(s), not {w.num_prompts}")
            if byte_checks != pulls:
                problems.append(f"{at}decode logged {byte_checks} byte check(s) for {pulls} distinct "
                                f"request(s){' that pulled' if prefix else ''}")
        for role, label in (("prefill", "prefill DP ranks produced"), ("decode", "decode DP ranks consumed")):
            shared = sorted(request for request, dps in owners[role].items() if len(dps) > 1)
            if shared:
                problems.append(f"{at}{len(shared)} request(s) more than one of the {label}: {shared[0]}")
        unpaired = sorted(set(owners["prefill"]) ^ set(owners["decode"]))
        if unpaired:
            problems.append(f"{at}{len(unpaired)} request(s) not both produced and consumed: {unpaired[0]}")
        owners_by_tp.append(owners)
        handoff = None
        if any(summary["hybrid"] for summary in of["decode"].values()):
            # The requests decode consumed: its byte checks, else its
            # allocations (never its steps alone).
            consumed_ids = "checked_ids" if run.kv_check else "allocated_ids"
            handoff = _handoff_from_sources(
                joint_steps, tp, hetero["sources"][tp] if hetero else [tp],
                set().union(*(summary[consumed_ids] for summary in of["decode"].values())))
            problems.extend(f"{at}{problem}" for problem in handoff["problems"])
        hit = {role: {key: sum(summary[key] for summary in of[role].values())
                      for key in ("kv_hit_checks", "kv_hit_blocks", "kv_bytes_hit_checked")} for role in sides}
        if prefix:
            for role in sides:
                if not (hit[role]["kv_hit_blocks"] and hit[role]["kv_bytes_hit_checked"]):
                    problems.append(f"{at}{role} showed no checked prefix hit: the run did not exercise prefix "
                                    "caching")
        fragmentation = _fragmentation(
            [records for (dp, rank), records in logs["prefill"].items() if rank == first],
            [records for (dp, rank), records in logs["decode"].items() if rank == tp],
            [ranks["decode"][(dp, rank)]["verified_ids"] for (dp, rank) in logs["decode"] if rank == tp])
        if tp == 0 and run.expect_fragmentation and not fragmentation["fragmented_both"]:
            problems.append(f"{at}no byte-checked transfer was fragmented on both instances: {fragmentation}")
        per_tp.append({
            "kv_checks": byte_checks, "consumed": consumed, "kv_bytes_checked": checked, "kv_bytes_spanned": spanned,
            "prefix_hits": hit if prefix else None, "fragmentation": fragmentation,
            "kv_bytes_transfer_only": sum(summary["kv_bytes_transfer_only"] for summary in of["decode"].values()),
            "transfers_bytes": moved,
            "kv_layout": {role: (of[role].get(0) or {}).get("kv_layout") for role in sides},
            "groups": {role: {dp: summary["groups"] for dp, summary in of[role].items()}
                       for role in sides} if any(s["groups"] for r in sides for s in of[r].values()) else None,
            "handoff": handoff and handoff["evidence"],
        })
    # Every TP rank of a request's owner is the same DP rank.
    for role, kind in (("prefill", "producers"), ("decode", "consumers")):
        first = owners_by_tp[0][role] if owners_by_tp else {}
        for tp, owners in enumerate(owners_by_tp[1:], start=1):
            differ = sorted(request for request in first if owners[role].get(request) != first[request])
            if differ:
                problems.append(f"rank {tp}: {len(differ)} request(s) with other {role} DP ranks than rank 0's")
        for dps in first.values():
            for dp in dps:
                ownership[kind][dp] = ownership[kind].get(dp, 0) + 1

    if routers is not None:
        # Every KV byte crosses its TP rank's network's router once, and no
        # other: each carries its ranks' bytes and little more (headers,
        # acknowledgements).
        expected: dict[str, int] = {}
        # One side's connector counts them: Mooncake's prefill, NIXL's decode.
        for role in sides:
            for (dp, tp), summary in ranks[role].items():
                network = sides[role].rank_networks(sizes[role])[tp]
                expected[network] = expected.get(network, 0) + summary["transfers"]["bytes"]
        for network, carried_by_ranks in expected.items():
            carried = routers.get(network, 0)
            if carried < carried_by_ranks:
                problems.append(f"the KV network {network} carried {carried} bytes, fewer than the {carried_by_ranks} "
                                "its ranks moved")
            elif carried > carried_by_ranks * 1.05 + (1 << 20):
                problems.append(f"the KV network {network} carried {carried} bytes, more than its ranks' "
                                f"{carried_by_ranks} (another rank's KV, or traffic the connectors did not count)")
        moved = sum(summary["transfers"]["bytes"] for role in sides for summary in ranks[role].values())
        control = routers.get(run.control_network, 0)
        if moved and control > moved / 10:
            problems.append(f"the control network carried {control} bytes, too much beside {moved} of KV")
    sums = {key: sum(summary["transfers"][key] for role in sides for summary in ranks[role].values())
            for key in ("count", "bytes", "failed")}
    speculation = None
    if run.server.speculative is not None:
        speculation = _speculation(bench, logs["decode"], run)
        problems.extend(speculation.pop("problems"))
    return {
        **({"transport": {"backend": nixl_backend(run.server), "tent_applicable": False}} if not uses_tent(run.server) else {}),
        "problems": problems,
        "requests": bench.get("completed"),
        "kv_checks": per_tp[0]["kv_checks"] if per_tp else 0,
        "kv_bytes_checked": sum(rank["kv_bytes_checked"] for rank in per_tp),
        "kv_bytes_transfer_only": sum(rank.get("kv_bytes_transfer_only", 0) for rank in per_tp),
        # TP rank 0's hit checks (every TP rank checks the same requests),
        # every rank's bytes.
        "prefix_hits": {role: {"kv_hit_checks": per_tp[0]["prefix_hits"][role]["kv_hit_checks"],
                               "kv_hit_blocks": per_tp[0]["prefix_hits"][role]["kv_hit_blocks"],
                               "kv_bytes_hit_checked": sum(rank["prefix_hits"][role]["kv_bytes_hit_checked"]
                                                           for rank in per_tp)}
                        for role in sides} if prefix and per_tp else None,
        "fragmentation": per_tp[0]["fragmentation"] if per_tp else None,
        "transfers": sums,
        "router_bytes": routers,
        "ranks": per_tp,
        "ownership": ownership,
        "dp_ranks": {role: [{"dp_rank": dp, "tp_rank": tp,
                             **{key: value for key, value in summary.items()
                                if key not in ("requests", "checked_ids", "verified_ids", "allocated_ids")},
                             "requests": len(summary["requests"])}
                            for (dp, tp), summary in ranks[role].items()] for role in sides},
        "speculation": speculation,
        "hetero_tp": hetero,
        "hp_tcp_task_limit": hp_tcp_task_limit(run.server),
        "ttft_ms": {key: bench.get(f"{key}_ttft_ms") for key in ("mean", "median", "p99")},
        "tpot_ms": {key: bench.get(f"{key}_tpot_ms") for key in ("mean", "median", "p99")},
    }



def _speculation(bench: dict, decode: dict[tuple[int, int], list[dict]], run: Run) -> dict:
    """Speculative decoding's evidence: every request returned exactly its
    output length (a last step's surplus drafts clipped); decode scheduled
    drafts, and each of a step's drafted requests has exactly one acceptance
    record (`spec_acceptance`) in its owning DP rank's log, for that step:
    its drafts, and accepted drafts within them, the step's own count where
    it has one (a joint MoE-DP step does not); all of them or none accepted
    at rates all 1 or all 0. TP ranks log the same: TP rank 0's logs."""

    spec = run.server.speculative
    problems = []
    lengths = bench.get("output_lens") or []
    if len(lengths) != run.workload.num_prompts or any(n != run.workload.output_len for n in lengths):
        problems.append(f"output lengths {sorted(set(lengths))} of {len(lengths)} request(s), not "
                        f"{run.workload.output_len} each")

    def count(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    # (owning DP rank, step, request): its drafts and the step's own count.
    scheduled: dict[tuple[int, int, str], tuple[int, int | None]] = {}
    logged: dict[tuple[int, int, str], list[object]] = {}
    duplicates, invalid = [], []
    steps = 0
    for (dp, tp), records in decode.items():
        if tp:
            continue
        for record in records:
            if "step" in record:
                step = record["step"]
                requests = step["requests"]
                drafts = [r["num_spec_tokens"] for r in requests]
                steps += any(drafts)
                counts = (record.get("cost") or {}).get("num_accepted")
                if counts is not None and any(drafts) and (
                        not isinstance(counts, list) or len(counts) != len(drafts)
                        or not all(count(a) and a <= d for a, d in zip(counts, drafts))):
                    problems.append(f"decode step {step['step_index']} accepted {counts} of drafts {drafts}")
                    counts = None
                for n, (request, d) in enumerate(zip(requests, drafts)):
                    if d:
                        owner = request.get("dp_rank", dp)
                        scheduled[(owner, step["step_index"], request["req_id"])] = (
                            d, counts[n] if counts is not None else None)
            if "spec_acceptance" in record:
                entry = record["spec_acceptance"]
                for request, pair in (entry.get("requests") or {}).items():
                    key = (dp, entry.get("step_index"), request)
                    if not isinstance(pair, list) or len(pair) != 2 or not all(map(count, pair)) or (
                            not 0 < pair[0] or pair[1] > pair[0]):
                        invalid.append(f"DP rank {dp} step {key[1]}: request {request} accepted {pair} "
                                       "(drafts, accepted)")
                    elif key in logged:
                        duplicates.append(f"DP rank {dp} step {key[1]}: request {request} logged twice")
                    else:
                        logged[key] = pair
    problems.extend(_capped(invalid) + _capped(duplicates))
    missing = [key for key in scheduled if key not in logged]
    extra = [key for key in logged if key not in scheduled]
    contradicted = [
        f"DP rank {key[0]} step {key[1]}: request {key[2]} logged {logged[key]}, the step drafted {d}"
        + (f" and accepted {a}" if a is not None else "")
        for key, (d, a) in scheduled.items() if key in logged
        and (logged[key][0] != d or (a is not None and logged[key][1] != a))]
    drafted = sum(d for d, _ in scheduled.values())
    accepted = sum(pair[1] for key, pair in logged.items() if key in scheduled)
    if not drafted:
        problems.append("decode scheduled no drafts")
    if missing:
        problems.append(f"{len(missing)} drafted request step(s) lack their owner's acceptance (spec_acceptance), "
                        f"e.g. DP rank {missing[0][0]} step {missing[0][1]} request {missing[0][2]}")
    if extra:
        problems.append(f"{len(extra)} acceptance record(s) for no drafted request step of their DP rank, e.g. "
                        f"DP rank {extra[0][0]} step {extra[0][1]} request {extra[0][2]}")
    problems.extend(_capped(contradicted))
    rates = spec["acceptance_rates"]
    if drafted and set(rates) == {1} and accepted != drafted:
        problems.append(f"{accepted} of {drafted} drafts accepted at acceptance rates 1")
    elif set(rates) == {0} and accepted:
        problems.append(f"{accepted} drafts accepted at acceptance rates 0")
    by_dp: dict[str, dict[str, int]] = {}
    for (owner, _, _), pair in sorted(logged.items(), key=lambda item: str(item[0])):
        rank = by_dp.setdefault(str(owner), {"drafted": 0, "accepted": 0})
        rank["drafted"] += pair[0]
        rank["accepted"] += pair[1]
    return {"problems": problems, "num_speculative_tokens": spec["num_speculative_tokens"],
            "acceptance_rates": rates, "seed": spec.get("seed", 0), "steps_with_drafts": steps,
            "drafted": drafted, "drafts_logged": sum(pair[0] for pair in logged.values()), "accepted": accepted,
            "dp_ranks": by_dp}



def _pull_plans(records: list[dict]) -> dict[str, dict[int, tuple[int, int]]]:
    """Decode's pulls as the connector posted them: for each request in a
    step's ``reads_posted``, that step's ``allocated_groups`` entries (the
    blocks it pulls, not the request's later scheduled table), as each
    group's first place and blocks holding KV."""
    plans: dict[str, dict[int, tuple[int, int]]] = {}
    for record in records:
        events = record.get("kv_events") or {}
        by_group = events.get("allocated_groups") or {}
        for request in events.get("reads_posted") or []:
            plans[request] = {entry["group"]: (entry["first"], sum(1 for block in entry["blocks"] if block))
                              for entry in by_group.get(request, [])}
    return plans



def _prefill_backoff(layout: dict) -> bool:
    """Whether decode loads N - 1 of a prompt's N tokens and computes the
    last: a recurrent state's, or (``prefill_backoff``) a window replayed
    after a load (DeepSeek-V4's)."""
    return layout.get("prefill_backoff") == 1 or any(
        group.get("kind") == "MambaSpec" for group in layout.get("groups") or [])



def _packed_problems(group: dict) -> list[str]:
    """How a packed group's layout (each block all of its layers' pages,
    DeepSeek-V4's) is malformed: each member names its kind, dtype, tokens a
    state, layer count and one offset per layer, its real bytes within its
    page; every page lies within the block stride, none overlapping another;
    the members count the group's layers, its checked bytes are their real
    pages and the bytes a block moves span the block. Empty for a group that
    is not packed."""
    if "kinds" in group and "members" not in group:
        return _kinds_problems(group)
    if not any(key in group for key in ("members", "checked_bytes_per_block", "transfer_bytes_per_block")):
        return []

    def size(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    members = group.get("members")
    checked = group.get("checked_bytes_per_block")
    moved = group.get("transfer_bytes_per_block")
    stride = group.get("block_stride")
    if not (isinstance(members, list) and members and size(checked) and size(moved) and size(stride)):
        return [f"has a malformed packed layout (members, checked_bytes_per_block {checked!r}, "
                f"transfer_bytes_per_block {moved!r}, block_stride {stride!r})"]
    problems, pages = [], []
    for index, member in enumerate(members):
        fields = ("kind", "dtype", "tokens_per_state", "layers", "offsets", "page_bytes", "real_page_bytes")
        if not isinstance(member, dict) or any(key not in member for key in fields):
            return [f"has a malformed packed member {index} (needs {', '.join(fields)})"]
        offsets, per_state = member["offsets"], member["tokens_per_state"]
        if not (isinstance(member["kind"], str) and isinstance(member["dtype"], str)
                and isinstance(per_state, (int, float)) and not isinstance(per_state, bool) and per_state > 0
                and size(member["layers"]) and size(member["page_bytes"]) and size(member["real_page_bytes"])
                and isinstance(offsets, list) and len(offsets) == member["layers"]
                and all(isinstance(o, int) and not isinstance(o, bool) and o >= 0 for o in offsets)):
            return [f"has a malformed packed member {index} ({member!r})"]
        if member["real_page_bytes"] > member["page_bytes"]:
            problems.append(f"member {index} has {member['real_page_bytes']} real bytes in a "
                            f"{member['page_bytes']}-byte page")
        for offset in offsets:
            if offset + member["page_bytes"] > stride:
                problems.append(f"member {index}'s page at {offset} ends beyond the {stride}-byte block")
            pages.append((offset, offset + member["page_bytes"], index))
    pages.sort()
    for (_, end, first), (start, _, second) in zip(pages, pages[1:]):
        if start < end:
            problems.append(f"member {second}'s page at {start} overlaps member {first}'s, ending at {end}")
    layers = sum(member["layers"] for member in members)
    real = sum(member["real_page_bytes"] * member["layers"] for member in members)
    if layers != group.get("layers"):
        problems.append(f"packs {layers} layers, not its {group.get('layers')}")
    if real != checked:
        problems.append(f"checks {checked} bytes a block, not its members' {real} real bytes")
    if moved < stride:
        problems.append(f"moves {moved} bytes a block, less than its {stride}-byte block")
    return problems



def _kinds_problems(group: dict) -> list[str]:
    """How a group of layers of several pages, each layer in its own region
    (a DSA layer's latent and indexer keys), is malformed: each kind names
    its role, dtype, layer count, page and real bytes and block stride, its
    real bytes within its page and its page within its block; the kinds
    count the group's layers and its checked bytes a block are their pages
    (not the group's first tensor's)."""
    kinds = group.get("kinds")
    checked = group.get("checked_bytes_per_block")
    fields = ("role", "dtype", "layers", "page_bytes", "real_page_bytes", "block_stride")

    def size(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    if not (isinstance(kinds, list) and len(kinds) > 1 and size(checked)
            and all(isinstance(k, dict) and all(f in k for f in fields) for k in kinds)
            and all(size(k["layers"]) and size(k["page_bytes"]) and size(k["real_page_bytes"])
                    and size(k["block_stride"]) for k in kinds)):
        return [f"has a malformed layout of several pages (kinds {kinds!r}, checked_bytes_per_block {checked!r})"]
    problems = []
    if sum(k["layers"] for k in kinds) != group.get("layers"):
        problems.append(f"has kinds of {sum(k['layers'] for k in kinds)} layers, not its {group.get('layers')}")
    if checked != sum(k["page_bytes"] * k["layers"] for k in kinds):
        problems.append(f"checks {checked} bytes a block, not its kinds' pages")
    problems.extend(f"kind {k['role']} has {k['real_page_bytes']} real bytes in a {k['page_bytes']}-byte page"
                    for k in kinds if k["real_page_bytes"] > k["page_bytes"])
    problems.extend(f"kind {k['role']}'s {k['page_bytes']}-byte page exceeds its {k['block_stride']}-byte block"
                    for k in kinds if k["page_bytes"] > k["block_stride"])
    return problems



def _spanned_bytes(layout: dict, checks: list[dict]) -> int:
    """The bytes a rank's successful byte checks span as the connector moves
    them: their checked bytes, but a packed group's blocks whole."""
    by_group = dict(enumerate(layout["groups"]))
    total = 0
    for check in checks:
        if "error" in check:
            continue
        for g in check["groups"]:
            group = by_group.get(g.get("group"), {})
            moved = group.get("transfer_bytes_per_block")
            packed = isinstance(moved, int) and not _packed_problems(group)
            total += g["blocks"] * moved if packed else int(g.get("bytes") or 0)
    return total



def _byte_check_problems(layout: dict, checks: list[dict], input_len: int,
                         plans: dict[str, dict[int, tuple[int, int]]],
                         prompts: dict[str, int] | None = None, prefix: bool = False) -> list[str]:
    """How a rank's successful byte checks disagree with its KV layout.

    Each check has blocks, bytes and nested results covering each KV cache group
    once, its blocks and bytes their sums; each group holds blocks (with
    prefix caching off a pull has some of every group, but a 1-token
    window's of a block-aligned prompt, all skipped), its bytes being
    its blocks times its layers' block bytes; a window pulls its last
    blocks only (Mooncake's cdiv(window, block) + 1), other attention from
    the first, all of the request's prompt (`prompts`, by decode's request
    id): cdiv(tokens, block) blocks of its N tokens, N - 1 when the
    decoder computes the last token (a hybrid model's, or a layout's
    ``prefill_backoff``); a gated delta-net's state
    is one block, at place 0. For several groups or a window, each group
    must also be the request's pull as posted (``_pull_plans``), and prompts
    longer than a window must have had some pull clipped. With prefix
    caching (`prefix`) a pull starts after decode's own prefix hit, any
    block, and takes the rest of the prompt's. A packed group
    (each block all of its layers' pages, DeepSeek-V4's) checks its layers'
    real pages, ``checked_bytes_per_block`` a block. Missing fields are
    problems."""
    groups = layout["groups"]
    problems = []
    grouped = len(groups) > 1 or any(spec.get("sliding_window") is not None for spec in groups)
    # The decoder computes the last of N > 1 prompt tokens (a recurrent
    # state, or a window replayed after a load).
    hybrid = _prefill_backoff(layout)
    if layout.get("prefill_backoff") not in (None, 1):
        problems.append(f"the layout's prefill_backoff is {layout.get('prefill_backoff')!r}, not 1")
    invalid = set()
    for index, spec in enumerate(groups):
        if found := _packed_problems(spec):
            invalid.add(index)
            problems.extend(f"group {index} {problem}" for problem in found)
    clipped = [0] * len(groups)
    for check in checks:
        request = check.get("req_id")
        if missing := [key for key in ("blocks", "bytes", "groups") if key not in check]:
            problems.append(f"{request}'s byte check lacks {', '.join(missing)}")
            continue
        nested = check["groups"]
        ids = sorted(g["group"] for g in nested)
        if ids != list(range(len(groups))):
            problems.append(f"{request}'s byte check covers groups {ids}, not each of the layout's "
                            f"{len(groups)} once")
        if (check["blocks"], check["bytes"]) != (sum(g["blocks"] for g in nested), sum(g["bytes"] for g in nested)):
            problems.append(f"{request}'s byte check is not its groups' sum")
        # With prefix caching a pull's plan is logged for one group too.
        plan = plans.get(request) if grouped or prefix else None
        if (grouped or prefix) and plan is None:
            problems.append(f"{request}'s pull plan was not logged (reads_posted with allocated_groups)")
        for g in nested:
            spec = groups[g["group"]] if 0 <= g["group"] < len(groups) else None
            if spec is None:
                continue
            window = spec.get("sliding_window")
            prompt = (prompts or {}).get(request)
            # A hybrid's decoder computes the last of N > 1 tokens; one
            # token's pages are pulled whole, the token computed again.
            tokens = prompt - 1 if hybrid and prompt is not None and prompt > 1 else prompt
            # The next token's 1-token window skips a block-aligned prompt whole.
            skipped = window == 1 and tokens is not None and tokens % spec["block_size"] == 0
            if skipped and (g["blocks"], g["first_logical_block"]) != (0, tokens // spec["block_size"]):
                problems.append(f"{request}'s 1-token window group {g['group']} checked {g['blocks']} blocks from "
                                f"{g['first_logical_block']}, not 0 from {tokens // spec['block_size']} (a "
                                f"block-aligned prompt's end)")
            elif g["blocks"] < 1 and not skipped:
                problems.append(f"{request}'s group {g['group']} checked no blocks")
            if g["group"] in invalid:
                pass  # its malformed layout is reported above
            elif "transfer" in spec:
                # Several groups over NIXL: the native selection of each
                # block, its transfer-only regions and its state pages'
                # unmoved padding.
                transfer = spec["transfer"]
                want = {key: g["blocks"] * transfer[per] for key, per in (
                    ("bytes", "bytes_per_block"), ("transfer_only_bytes", "transfer_only_bytes_per_block"),
                    ("padding_bytes", "padding_bytes_per_block"), ("descriptors", "descriptors_per_block"))}
                got = {key: g.get(key) for key in want}
                if (g["layers"], got) != (spec["layers"], want):
                    problems.append(f"{request}'s group {g['group']} check disagrees with NIXL's selection "
                                    f"({spec['layers']} layers, {want}; checked {g['layers']} layers, {got})")
            elif "checked_bytes_per_block" in spec:
                # A packed block: its layers' real pages are checked.
                if (g["layers"], g["bytes"]) != (spec["layers"], g["blocks"] * spec["checked_bytes_per_block"]):
                    problems.append(f"{request}'s group {g['group']} check disagrees with the layout "
                                    f"({spec['layers']} layers, {spec['checked_bytes_per_block']} bytes checked "
                                    "a block)")
            elif (g["layers"], g["bytes"]) != (spec["layers"], g["blocks"] * spec["layers"] * spec["block_bytes"]):
                problems.append(f"{request}'s group {g['group']} check disagrees with the layout "
                                f"({spec['layers']} layers of {spec['block_bytes']} bytes a block)")
            # A state's running slot (before its native MTP's scratch
            # slots, the one NIXL moves) has clipped-read metadata index k
            # (`speculative_blocks`); any other group starts at 0.
            start = (spec.get("speculative_blocks") or 0) if spec.get("kind") == "MambaSpec" else 0
            if window is None and g["first_logical_block"] != start and not prefix:
                problems.append(f"{request}'s group {g['group']} ({spec.get('kind')}) was pulled from block "
                                f"{g['first_logical_block']}, not {start}")
            if spec.get("kind") == "MambaSpec":
                if g["blocks"] != 1:
                    problems.append(f"{request}'s state group {g['group']} checked {g['blocks']} blocks, not 1")
            elif window is None and prompt is None:
                problems.append(f"{request}'s group {g['group']} pages cannot be checked: decode logged no "
                                "steps of it")
            elif window is None:
                hit = g["first_logical_block"] if prefix else 0
                if g["blocks"] != -(-tokens // spec["block_size"]) - hit:
                    problems.append(f"{request}'s group {g['group']} checked {g['blocks']} blocks, not the "
                                    f"cdiv({tokens}, {spec['block_size']}) of its prompt"
                                    + (f" past its {hit}-block prefix hit" if hit else ""))
            if window is not None:
                bound = -(-window // spec["block_size"]) + 1
                if g["blocks"] > bound:
                    problems.append(f"{request}'s window group {g['group']} pulled {g['blocks']} blocks, "
                                    f"more than its window's {bound}")
                clipped[g["group"]] += g["first_logical_block"] > 0
            # A pull posts no entry for a group it pulls nothing of.
            if plan is not None and plan.get(g["group"], (tokens // spec["block_size"], 0) if skipped else None) != (
                    g["first_logical_block"], g["blocks"]):
                first, blocks = plan.get(g["group"]) or (None, 0)
                problems.append(f"{request}'s group {g['group']} checked {g['blocks']} block(s) from "
                                f"{g['first_logical_block']}, not the pull's {blocks} from {first}")
    for index, spec in enumerate(groups):
        window = spec.get("sliding_window")
        if checks and window is not None and input_len > window + 2 * spec["block_size"] and not clipped[index]:
            problems.append(f"no pull of window group {index} was clipped, with prompts of {input_len} tokens "
                            f"beyond its {window}")
    return _capped(problems)



def _runs(blocks: list[int]) -> int:
    """Physically consecutive runs of a block list."""
    return len(blocks) and 1 + sum(b != a + 1 for a, b in zip(blocks, blocks[1:]))



def _fragmentation(prefill: list[list[dict]], decode: list[list[dict]], verified: list[set[str]]) -> dict:
    """How scattered one TP rank's pulled blocks were: for each pull decode
    posted, its blocks of each group (``allocated_groups`` of that step) and
    the producer's at the same places (its allocations, by base request id),
    each side's physical runs. A pull is fragmented on a side when a group's
    blocks there are not one run; jointly contiguous runs (consecutive on
    both sides at once) are what a connector can merge into one transfer.
    Only pulls with a successful byte check (`verified[i]`, decode's i-th
    log's) whose every block has the producer's are classified; the others
    are counted apart: unlogged (no allocations by group: one full-attention
    group without prefix caching), unchecked, unmatched (a producer place
    missing; `unmatched_groups`, their groups)."""
    produced: dict[str, dict[int, dict[int, int]]] = {}
    for records in prefill:
        for record in records:
            for request, entries in ((record.get("kv_events") or {}).get("allocated_groups") or {}).items():
                table = produced.setdefault(_base_request_id(request), {})
                for entry in entries:
                    places = table.setdefault(entry["group"], {})
                    for offset, block in enumerate(entry["blocks"]):
                        places[entry["first"] + offset] = block
    kinds = {(False, False): "contiguous", (True, False): "fragmented_prefill_only",
             (False, True): "fragmented_decode_only", (True, True): "fragmented_both"}
    counts = dict.fromkeys(kinds.values(), 0)
    joint = unlogged = unchecked = unmatched = unmatched_groups = 0
    for records, checked in zip(decode, verified):
        for record in records:
            events = record.get("kv_events") or {}
            for request in events.get("reads_posted") or []:
                table = produced.get(_base_request_id(request)) or {}
                entries = (events.get("allocated_groups") or {}).get(request)
                if not entries:
                    unlogged += 1
                    continue
                if request not in checked:
                    unchecked += 1
                    continue
                groups = [[(block, (table.get(entry["group"]) or {}).get(entry["first"] + offset))
                           for offset, block in enumerate(entry["blocks"]) if block] for entry in entries]
                missing = sum(any(p is None for _, p in pairs) for pairs in groups)
                if missing:
                    unmatched += 1
                    unmatched_groups += missing
                    continue
                sides = [False, False]
                for pairs in groups:
                    ours, theirs = [d for d, _ in pairs], [p for _, p in pairs]
                    sides[0] |= _runs(theirs) > 1
                    sides[1] |= _runs(ours) > 1
                    joint += len(pairs) and 1 + sum(
                        not (d2 == d1 + 1 and p2 == p1 + 1) for (d1, p1), (d2, p2) in zip(pairs, pairs[1:]))
                counts[kinds[tuple(sides)]] += 1
    return {"pulls": sum(counts.values()) + unlogged + unchecked + unmatched, **counts, "unlogged": unlogged,
            "unchecked": unchecked, "unmatched": unmatched, "jointly_contiguous_runs": joint,
            "unmatched_groups": unmatched_groups}



def kv_heads_of(model: Path) -> int | None:
    """The model's KV heads as its KV cache holds them, from its own config
    (a multimodal model's text config): an MLA model's (`kv_lora_rank`) one
    latent, the same on every rank, whatever attention heads it computes."""
    try:
        config = json.loads((Path(model) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    sources = (config, config.get("text_config") or {}, config.get("llm_config") or {})
    if any(source.get("kv_lora_rank") for source in sources):
        return 1
    for source in sources:
        if isinstance(source.get("num_key_value_heads"), int):
            return source["num_key_value_heads"]
    return None



def global_heads(rank: int, tp: int, kv_heads: int) -> list[int]:
    """The global KV heads vLLM gives TP rank `rank` of `tp`: K/tp of them
    from rank * K/tp, or with tp > K one, head rank * K/tp (replicated)."""
    if tp <= kv_heads:
        return list(range(rank * kv_heads // tp, (rank + 1) * kv_heads // tp))
    return [rank * kv_heads // tp]



def group_kv_heads(model: Path, kinds: list) -> tuple[list[int] | None, str | None]:
    """Each KV cache group's total KV heads, from its layout `kind`: the
    model's own for all groups, except in the config family that gives its
    full-attention layers a count of their own (`num_global_key_value_heads`,
    Gemma4): a `FullAttentionSpec` group takes that, a `SlidingWindowSpec`
    group `num_key_value_heads`. Any other kind in such a config, or an
    invalid count, is an error: never a default. (None, error) then."""
    uniform = kv_heads_of(model)
    if uniform is None:
        return None, f"{model}/config.json gives no num_key_value_heads"
    try:
        config = json.loads((Path(model) / "config.json").read_text())
    except (OSError, ValueError):
        return None, f"{model}/config.json is unreadable"
    sources = (config, config.get("text_config") or {}, config.get("llm_config") or {})
    own = next((source["num_global_key_value_heads"] for source in sources
                if source.get("num_global_key_value_heads") is not None), None)
    if own is None:
        return [uniform] * len(kinds), None
    if isinstance(own, bool) or not isinstance(own, int) or own <= 0:
        return None, f"num_global_key_value_heads {own!r} is not a positive integer"
    totals = []
    for kind in kinds:
        if kind == "FullAttentionSpec":
            totals.append(own)
        elif kind == "SlidingWindowSpec":
            totals.append(uniform)
        else:
            return None, (f"a {kind} group has no KV head rule in a config with num_global_key_value_heads "
                          "(full-attention and sliding-window groups only)")
    return totals, None



def _hetero_tp(run: Run, sizes: dict[str, int], logs: dict, ranks: dict, ports: dict | None,
               unwritten: set[int] = frozenset()) -> dict:
    """Unequal prefill and decode TP sizes, reconciled with the native
    pairing computed here: each decode rank's sources (decode TP >= prefill
    TP: rank d * P/D; else ranks d * P/D + i), each of its global heads sent
    by the lowest-ranked source holding it (a replicated producer's other
    ranks send nothing: Mooncake's answer without a payload, for each of
    the decode rank's requests); every rank's layout and decode check
    holding its own global heads; a head's bytes in a block of a layer (its
    slab) from each side's layout, the same on both; each pair's logged
    payload equal to the heads it carries; each Mooncake write's plan at
    those heads' places, in slabs; each NIXL READ's measured descriptors
    (as many as its hook saw, within NIXL's limit) each one layer's block of
    the heads it carries; each rank's router port carrying its own payload
    (prefill RX, decode TX); unique KV apart from the replicated payload.
    Producer ranks without a TENT write series (`unwritten`) must be such
    idle replicas."""

    problems: list[str] = []
    prefill_tp, decode_tp = sizes["prefill"], sizes["decode"]
    nixl = run.server.connector == "nixl"
    kv_heads = kv_heads_of(run.model)
    evidence: dict = {"sizes": sizes, "kv_heads": kv_heads}
    if kv_heads is None:
        return {**evidence, "problems": [f"{run.model}/config.json gives no num_key_value_heads"],
                "sources": {d: [0] for d in range(decode_tp)}, "received": {}}
    def layout(role: str, rank: int) -> dict | None:
        return next((r["kv_layout"] for r in logs[role].get((0, rank), []) if "kv_layout" in r), None)

    # Each group's kind, the same on every rank of both sides (a group's
    # correspondence across P and D), then its total KV heads.
    kind_lists = {(role, rank): [g.get("kind") for g in (layout(role, rank) or {}).get("groups") or []]
                  for role in sizes for rank in range(sizes[role])}
    kinds = next((k for k in kind_lists.values() if k), [])
    for (role, rank), found in sorted(kind_lists.items()):
        if found != kinds:
            problems.append(f"{role} rank {rank}'s groups are {found}, not the others' {kinds}")
    if "MambaSpec" in kinds:
        return _hetero_hybrid(run, sizes, logs, ranks, ports, unwritten, kinds, kv_heads, evidence, problems, layout)
    totals, why = group_kv_heads(run.model, kinds) if kinds else ([kv_heads], None)
    if why:
        return {**evidence, "problems": problems + [why], "sources": {d: [0] for d in range(decode_tp)},
                "received": {}}
    evidence["group_kv_heads"] = totals
    gheads = {role: {rank: [global_heads(rank, sizes[role], k) for k in totals] for rank in range(sizes[role])}
              for role in sizes}
    heads = {role: {rank: gheads[role][rank][0] for rank in gheads[role]} for role in sizes}
    ratio = max(prefill_tp, decode_tp) // min(prefill_tp, decode_tp)
    paired = {d: [d * prefill_tp // decode_tp] if decode_tp >= prefill_tp else
              [d * ratio + i for i in range(ratio)] for d in range(decode_tp)}
    gsender = [{(h, d): min(p for p in paired[d] if h in gheads["prefill"][p][g])
                for d in range(decode_tp) for h in gheads["decode"][d][g]} for g in range(len(totals))]
    sender = gsender[0]
    # NIXL reads each unique head once (its first rank); Mooncake's decode
    # pulls every paired producer.
    sources = {d: sorted({gsender[g][(h, d)] for g in range(len(totals)) for h in gheads["decode"][d][g]})
               if nixl else paired[d] for d in range(decode_tp)}
    evidence["heads"] = heads
    evidence["sources"] = sources

    # Each group's head slab: its page (padding excluded) over its heads,
    # the same on every rank of both sides.
    slabs: dict[tuple[str, int], tuple[int | None, ...]] = {}
    for role in sizes:
        for rank in range(sizes[role]):
            groups = (layout(role, rank) or {}).get("groups") or []
            mine_slabs = []
            for index, group in enumerate(groups):
                want = gheads[role][rank][index] if index < len(totals) else None
                if want is None or group.get("pattern") != "head-v1" or group.get("global_heads") != want:
                    problems.append(f"{role} rank {rank}'s group {index} holds {group.get('pattern')} heads "
                                    f"{group.get('global_heads')}, not head-v1 heads {want}")
                    mine_slabs.append(None)
                    continue
                page = group.get("real_page_bytes") or group.get("block_bytes")
                mine_slabs.append(page // len(want) if (
                    isinstance(page, int) and page > 0 and page % len(want) == 0) else None)
            slabs[(role, rank)] = tuple(mine_slabs) or (None,)
    distinct = set(slabs.values())
    gslabs = list(next(iter(distinct))) if len(distinct) == 1 else [None]
    if None in gslabs or len(distinct) != 1:
        problems.append(f"the ranks' layouts give no single head size in a block of a layer: "
                        f"{ {f'{role}{rank}': (n[0] if len(set(n)) == 1 else n) for (role, rank), n in sorted(slabs.items())} }")
        gslabs = [None] * len(totals)
    slab = gslabs[0] if len(set(gslabs)) == 1 else gslabs
    evidence["slab_bytes"] = slab

    expected: dict[tuple[int, int], int] = {}
    unique = 0
    for d in range(decode_tp):
        mine = heads["decode"][d]
        for record in logs["decode"].get((0, d), []):
            check = record.get("kv_check")
            if not check or check.get("kind") not in (None, "transfer") or "error" in check:
                continue
            cgroups = check.get("groups") or []
            for position, group in enumerate(cgroups):
                index = group.get("group", position)
                want = gheads["decode"][d][index] if isinstance(index, int) and 0 <= index < len(totals) else None
                if want is None or group.get("pattern") != "head-v1" or group.get("global_heads") != want:
                    problems.append(f"decode rank {d} checked {check.get('req_id')}'s heads "
                                    f"{group.get('global_heads')} ({group.get('pattern')}), not its {want}")
            # Bytes per head, separately for each KV cache group.
            per_group = []
            for group in cgroups:
                index = group["group"]
                per_group.append((index, group["bytes"] // len(gheads["decode"][d][index])))
            for index, per_head in per_group:
                if d == 0:
                    # Each request's KV once: every global head's bytes.
                    unique += per_head * totals[index]
                for p in {gsender[index][(h, d)] for h in gheads["decode"][d][index]}:
                    expected[(p, d)] = expected.get((p, d), 0) + per_head * sum(
                        1 for h in gheads["decode"][d][index] if gsender[index][(h, d)] == p)

    # Each decode rank's requests (its byte checks'), which each paired
    # producer answers.
    requested = {d: {_base_request_id(r["kv_check"]["req_id"]) for r in logs["decode"].get((0, d), [])
                     if "req_id" in (r.get("kv_check") or {}) and r["kv_check"].get("kind") in (None, "transfer")}
                 for d in range(decode_tp)}
    logged: dict[tuple[int, int], int] = {}
    answers: dict[tuple[int, int], list[str]] = {}
    # Producer ranks with an answer the pairing or its plan refutes.
    misanswered: set[int] = set()
    descriptors: list[int] = []
    merged: list[int] = []
    unknown = 0
    for role, mover in (("prefill", not nixl), ("decode", nixl)):
        if not mover:
            continue
        direction = "load" if nixl else "send"
        for (_, rank), records in logs[role].items():
            for record in records:
                transfer = (record.get("kv_transfers") or {}).get(direction) or {}
                for entry in transfer.get("each") or []:
                    peer, plan = entry[5:7]
                    if peer is None:
                        unknown += 1
                        continue
                    pair = (peer, rank) if nixl else (rank, peer)
                    logged[pair] = logged.get(pair, 0) + int(entry[0])
                    if nixl and isinstance(plan, list) and len(plan) > 1 and isinstance(plan[1], int):
                        descriptors.append(plan[1])
                    if nixl and isinstance(entry[3], int):
                        merged.append(entry[3])
                    problems.extend(_plan_problems(pair, entry, plan, nixl, gheads, gsender, sources, gslabs, descriptor_limit(run.server)))
                for answer in transfer.get("zero_payload") or []:
                    peer, requests, plan = answer
                    if not isinstance(peer, int) or rank not in paired.get(peer, ()):
                        problems.append(f"prefill rank {rank} answered decode rank {peer}, not one it pairs with")
                        misanswered.add(rank)
                        continue
                    if not isinstance(requests, list):
                        problems.append(f"prefill rank {rank} answered decode rank {peer} without a payload "
                                        f"for unnamed requests ({requests})")
                        misanswered.add(rank)
                        continue
                    answers.setdefault((rank, peer), []).extend(map(_base_request_id, requests))
                    if not plan or any(not isinstance(region, list) or region[0] for region in plan):
                        problems.append(f"prefill rank {rank} answered decode rank {peer} without a payload "
                                        f"under a plan that sends (or none): {plan}")
                        misanswered.add(rank)
    if unknown:
        problems.append(f"{unknown} transfer(s) logged without their peer rank")
    for pair in sorted(set(expected) | set(logged)):
        if logged.get(pair, 0) != expected.get(pair, 0):
            problems.append(f"prefill rank {pair[0]} -> decode rank {pair[1]} moved {logged.get(pair, 0)} bytes, "
                            f"not the {expected.get(pair, 0)} of the heads it holds for it")
    idle: set[int] = set()
    if not nixl:
        # Paired producers that hold no head they send: a replica's answer
        # for each of the decode rank's requests, once.
        covered = {p: p not in misanswered for p in range(prefill_tp)}
        for d in range(decode_tp):
            for p in paired[d]:
                answered = answers.get((p, d), [])
                if (p, d) in expected:
                    if answered:
                        problems.append(f"prefill rank {p} answered decode rank {d} without a payload for "
                                        f"{len(answered)} request(s), though it holds heads it sends")
                    covered[p] = False
                    continue
                twice = sorted({r for r in answered if answered.count(r) > 1})
                missing = sorted(requested[d] - set(answered))
                extra = sorted(set(answered) - requested[d])
                for what, which in (
                        (f"answered decode rank {d} without a payload twice for", twice),
                        (f"never answered decode rank {d} (no payload and no zero-payload answer) for", missing),
                        (f"answered decode rank {d} without a payload for requests it never pulled:", extra)):
                    if which:
                        problems.append(f"prefill rank {p} {what} {len(which)} request(s), {which[0]}")
                if twice or missing or extra or not requested[d]:
                    covered[p] = False
        idle = {p for p, ok in covered.items() if ok and not any(pp == p for pp, _ in logged)
                and any(p in paired[d] for d in range(decode_tp))}
    for p in sorted(unwritten - idle):
        problems.append(f"prefill rank {p}'s TENT metrics lack tent_write_bytes_total, and it is not a replica "
                        "whose zero-payload answers cover each of its decode ranks' requests")
    evidence["idle_replicas"] = sorted(idle)
    evidence["unwritten_accepted"] = sorted(unwritten & idle)
    if nixl and descriptor_limit(run.server) is not None and descriptors and max(descriptors) > descriptor_limit(run.server):
        problems.append(f"a NIXL READ selected {max(descriptors)} descriptors, more than the "
                        f"{NIXL_MAX_DESCRIPTORS} guard")

    received = {d: sum(n for (p, dd), n in logged.items() if dd == d) for d in range(decode_tp)}
    sent = {p: sum(n for (pp, d), n in logged.items() if pp == p) for p in range(prefill_tp)}
    payload = sum(logged.values())
    mover_role = "decode" if nixl else "prefill"
    moved = sum(summary["transfers"]["bytes"] for summary in ranks[mover_role].values())
    if moved != payload:
        problems.append(f"the connector moved {moved} bytes, not the {payload} its transfers' peers account for")
    for p in range(prefill_tp):
        produced = ranks["prefill"].get((0, p), {}).get("requests") or set()
        if run.kv_check and len(produced) != run.workload.num_prompts:
            problems.append(f"prefill rank {p} produced {len(produced)} request(s), not {run.workload.num_prompts}")

    port_evidence = {}
    if ports is not None:
        network = ports.get(run.prefill.kv_network) or {}
        by_rank = {(described["node"], described["slot"]): described for described in network.values()}
        for role, node, carried, direction in (("prefill", run.prefill.node, sent, "rx"),
                                               ("decode", run.decode.node, received, "tx")):
            for rank, payload_bytes in carried.items():
                port = by_rank.get((node, rank))
                if port is None:
                    problems.append(f"{role} rank {rank}'s router port was not measured")
                    continue
                measured = port[direction]
                port_evidence[f"{role}{rank}"] = {"rx": port["rx"], "tx": port["tx"], "payload": payload_bytes}
                if measured < payload_bytes or measured > payload_bytes * 1.05 + (1 << 20):
                    problems.append(f"{role} rank {rank}'s router port {direction.upper()} {measured} bytes, not its "
                                    f"{payload_bytes} of KV and little more")

    evidence.update({
        "pairs": [{"prefill": p, "decode": d, "bytes": logged.get((p, d), 0), "expected": expected.get((p, d), 0),
                   "zero_payload_answers": len(answers.get((p, d), []))}
                  for p in range(prefill_tp) for d in range(decode_tp)
                  if (p, d) in expected or (p, d) in logged or (p, d) in answers],
        "unique_kv_bytes": unique, "payload_bytes": payload,
        "replication_factor": round(payload / unique, 6) if unique else None,
        "max_descriptors": max(descriptors, default=None), "max_merged_descriptors": max(merged, default=None),
        "ports": port_evidence or None,
        "received": received, "problems": problems,
    })
    return evidence



def _plan_problems(pair: tuple[int, int], entry: list, plan: list | None, nixl: bool, heads: dict,
                   sender: dict, sources: dict, slab: int | None,
                   max_descriptors: int | None = NIXL_MAX_DESCRIPTORS) -> list[str]:
    """A transfer's logged plan against the heads its pair carries, in
    slabs (a head's bytes in a block of a layer, from the layouts). NIXL's
    [local slot, selected, predicted]: the slot the source's order gives;
    the descriptors vLLM selected (each one layer's block of the carried
    heads, so they give the bytes; within the guard) and the hook's
    prediction of the descriptors NIXL merges them into, which must be what
    NIXL measured (`entry[3]`), never more than selected; a READ without
    that prediction fails. Each of Mooncake's regions [sends,
    source offset, destination offset, length]: the first carried head's
    places in the producer's and decode's pages, the carried heads' length."""
    p, d = pair
    at = f"prefill rank {p} -> decode rank {d}: "
    # `heads`, `sender` and `slab` are each KV cache group's (lists by group).
    by_group = [[h for h in heads["decode"][d][g] if sender[g][(h, d)] == p] for g in range(len(sender))]
    carried = next((c for c in by_group if c), [])
    if not carried:
        return [f"{at}a transfer, though decode takes none of its heads from it"]
    known = all(n is not None for n in slab)
    spans = {len(c) * slab[g] for g, c in enumerate(by_group) if c and slab[g] is not None}
    if not plan:
        return [f"{at}a transfer of {entry[0]} bytes without its plan ({plan})"]
    if nixl:
        if not isinstance(plan, list) or len(plan) != 5:
            return [f"{at}a READ without its selected bytes and lengths ({plan})"]
        problems = []
        slot = sources[d].index(p) if p in sources[d] and len(sources[d]) > 1 else 0
        if plan[0] != slot:
            problems.append(f"{at}a READ into slot {plan[0]}, not {slot}")
        measured = entry[3]
        selected, predicted = plan[1:3]

        def count(value) -> bool:
            return isinstance(value, int) and not isinstance(value, bool) and value > 0

        if not count(selected):
            problems.append(f"{at}a READ without its selected descriptors ({selected})")
            return problems
        if not count(measured):
            problems.append(f"{at}a READ without NIXL's measured descriptors ({measured}; {selected} selected)")
        if not count(predicted):
            problems.append(f"{at}a READ without the hook's predicted merged descriptors ({predicted}; "
                            f"{selected} selected, NIXL measured {measured})")
        elif predicted > selected:
            problems.append(f"{at}a READ predicted at {predicted} merged descriptors, more than its {selected} "
                            "selected")
        if count(measured) and count(predicted) and measured != predicted:
            problems.append(f"{at}a READ NIXL measured at {measured} descriptors, predicted at {predicted} "
                            f"({selected} selected)")
        if count(measured) and measured > selected:
            problems.append(f"{at}a READ NIXL measured at {measured} descriptors, more than its {selected} "
                            "selected")
        if max_descriptors is not None and selected > max_descriptors:
            problems.append(f"{at}a READ of {selected} selected descriptors, more than the "
                            f"{NIXL_MAX_DESCRIPTORS} guard")
        if known and len(spans) != 1:
            problems.append(f"{at}a READ whose groups carry different spans {sorted(spans)}: the exact bytes "
                            "need one common carried span per descriptor")
        elif known:
            span = spans.pop()
            if int(entry[0]) != selected * span:
                if len({len(c) for c in by_group if c}) == 1 and len({n for n in slab}) == 1:
                    problems.append(f"{at}a READ of {entry[0]} bytes in {selected} selected descriptors, not "
                                    f"{len(carried)} head(s) of {slab[0]} bytes each")
                else:
                    problems.append(f"{at}a READ of {entry[0]} bytes in {selected} selected descriptors, not "
                                    f"{span} bytes (the carried span) each")
        return problems
    if not known:
        return []
    problems = []
    wants = []
    for g, c in enumerate(by_group):
        if c:
            want = [True, heads["prefill"][p][g].index(c[0]) * slab[g], heads["decode"][d][g].index(c[0]) * slab[g],
                    len(c) * slab[g]]
            if want not in wants:
                wants.append(want)
    for region in plan:
        if list(region) not in wants:
            shown = wants[0] if len(wants) == 1 else wants
            problems.append(f"{at}plan {list(region)}, not {shown} (heads {carried}, {slab[0]} bytes each)")
    return problems



def _state_spans(state: dict, rank: int, tp: int) -> list | None:
    """A rank's state spans [kind, component, offset, length, first, count]
    recomputed from the layout record's global components (the model's, the
    same on every rank): conv components in order, each its columns
    [rank*w/tp, ...) of `conv_rows * conv_itemsize` bytes, then the SSM
    heads. None when the layout does not split over tp."""
    column = state["conv_rows"] * state["conv_itemsize"]
    units = [w for _, w in state["components"]] + [state["ssm_heads"]]
    if tp < 1 or any(w % tp for w in units):
        return None
    spans, offset = [], 0
    for name, width in state["components"]:
        n = width // tp
        spans.append(["conv", name, offset, n * column, rank * n, n])
        offset += n * column
    n = state["ssm_heads"] // tp
    spans.append(["ssm", "ssm", offset, n * state["ssm_head_bytes"], rank * n, n])
    return spans



def _producers(rank: int, local_tp: int, remote_tp: int) -> list[int]:
    """The producers of a decode rank's pairing: one when the producer has
    no more ranks (floor(r*R/L)), else its R/L contiguous ones."""
    if remote_tp <= local_tp:
        return [rank * remote_tp // local_tp]
    k = remote_tp // local_tp
    return list(range(rank * k, (rank + 1) * k))



def _state_overlap(rank: int, local_tp: int, remote_tp: int, width: int) -> dict[int, tuple[int, int]]:
    """producer -> the global range [first, last) of a state component it
    supplies a decode rank: every candidate whose range overlaps the rank's."""
    lo, hi = rank * width // local_tp, (rank + 1) * width // local_tp
    out = {}
    for p in _producers(rank, local_tp, remote_tp):
        first, last = max(lo, p * width // remote_tp), min(hi, (p + 1) * width // remote_tp)
        if first < last:
            out[p] = (first, last)
    return out



def _hetero_hybrid(run: Run, sizes: dict[str, int], logs: dict, ranks: dict, ports: dict | None,
                   unwritten: set[int], kinds: list, kv_heads: int, evidence: dict, problems: list[str],
                   layout) -> dict:
    """Unequal TP sizes for a hybrid model over NIXL (its attention groups
    head-v1, its state groups state-v1), reconciled with ownership computed
    here: each state group's rank spans recomputed from the layout record's
    global components (never trusted), each decode rank's attention heads from
    the first candidate producer holding each (the union over heads) and its
    state ranges from every candidate whose range overlaps; each pair's bytes
    from the checks' per-group blocks, selected regions and spans (logical
    owned state, transfer-only regions and attention kept apart); each READ's
    exact bytes (its selected-byte sum and its length histogram) and the
    checker's per-READ record (producer, bytes by group, component and class,
    completion); NIXL's selected/predicted/measured descriptors. Mooncake does
    not pair a hybrid."""
    prefill_tp, decode_tp = sizes["prefill"], sizes["decode"]
    nixl = run.server.connector == "nixl"
    evidence["hybrid"] = True
    done = {"sources": {d: [0] for d in range(decode_tp)}, "received": {}}
    if not nixl:
        return {**evidence, "problems": problems + ["a hybrid model pairs unequal TP sizes over NIXL only"], **done}
    groups_of = lambda role, rank: (layout(role, rank) or {}).get("groups") or []
    state_idx = [i for i, k in enumerate(kinds) if k == "MambaSpec"]
    attn_idx = [i for i, k in enumerate(kinds) if k != "MambaSpec"]
    reference = None
    for role in sizes:
        for rank in range(sizes[role]):
            for index, group in enumerate(groups_of(role, rank)):
                if index in attn_idx:
                    want = global_heads(rank, sizes[role], kv_heads)
                    if group.get("pattern") != "head-v1" or group.get("global_heads") != want:
                        problems.append(f"{role} rank {rank}'s group {index} holds {group.get('pattern')} heads "
                                        f"{group.get('global_heads')}, not head-v1 heads {want}")
                    continue
                state = group.get("state")
                if group.get("pattern") != "state-v1" or not isinstance(state, dict):
                    problems.append(f"{role} rank {rank}'s group {index} is {group.get('pattern')}, not state-v1")
                    continue
                model = {k: state.get(k) for k in ("family", "components", "conv_rows", "conv_itemsize",
                                                   "ssm_heads", "ssm_head_bytes")}
                reference = reference or model
                if model != reference:
                    problems.append(f"{role} rank {rank}'s group {index} state layout {model} differs from {reference}")
                want_spans = _state_spans(reference, rank, sizes[role])
                if want_spans is None:
                    problems.append(f"{role} rank {rank}: the state does not split over TP {sizes[role]}")
                elif state.get("spans") != want_spans:
                    problems.append(f"{role} rank {rank}'s group {index} state spans {state.get('spans')}, "
                                    f"not the layout's {want_spans}")
    if reference is None or problems:
        return {**evidence, "problems": problems + ([] if reference else ["no state group's layout record"]), **done}
    evidence["state_layout"] = reference
    column = reference["conv_rows"] * reference["conv_itemsize"]

    # Attention sources: for each local head the first candidate holding it.
    def attention(d: int) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for h in global_heads(d, decode_tp, kv_heads):
            owner = next((q for q in _producers(d, decode_tp, prefill_tp)
                          if h in global_heads(q, prefill_tp, kv_heads)), None)
            if owner is None:
                problems.append(f"no producer of decode rank {d} holds head {h}")
            else:
                out.setdefault(owner, []).append(h)
        return out

    sources = {}
    for d in range(decode_tp):
        got = set(attention(d))
        for width in [w for _, w in reference["components"]] + [reference["ssm_heads"]]:
            got |= set(_state_overlap(d, decode_tp, prefill_tp, width))
        sources[d] = sorted(got)
    evidence["sources"] = sources
    expected: dict[tuple[int, int], int] = {}
    classes: dict[tuple[int, int, str], int] = {}
    attention_expected: dict[tuple[int, int], int] = {}
    totals = {"attention": 0, "state_logical": 0, "state_transfer_only": 0}
    unique = {"attention": 0, "state_logical": 0, "state_transfer_only": 0}
    checks: dict[int, list[dict]] = {d: [] for d in range(decode_tp)}
    for d in range(decode_tp):
        local = _state_spans(reference, d, decode_tp)
        span_bytes = sum(sp[3] for sp in local)
        for record in logs["decode"].get((0, d), []):
            check = record.get("kv_check")
            if not check or check.get("kind") not in (None, "transfer") or "error" in check:
                continue
            checks[d].append(check)
            coverage = check.get("coverage") or {}
            if not isinstance(coverage, dict) or "reads" not in coverage:
                problems.append(f"decode rank {d}'s check of {check.get('req_id')} has no READ coverage")
            for text in coverage.get("problems") or []:
                problems.append(f"decode rank {d} {check.get('req_id')}: {text}")
            if check.get("posted_matches_checked") is not True:
                problems.append(f"decode rank {d}'s check of {check.get('req_id')} did not match its posted READs")
            for read in coverage.get("reads") or []:
                if read.get("completed") is not True:
                    problems.append(f"decode rank {d}: a READ from {read.get('rank')} was not complete")
                if sum(c[1] for c in (read.get("classes") or {}).values()) != read.get("bytes"):
                    problems.append(f"decode rank {d}: a READ's classes sum to "
                                    f"{sum(c[1] for c in (read.get('classes') or {}).values())}, not {read.get('bytes')}")
            for position, group in enumerate(check.get("groups") or []):
                index = group.get("group", position)
                if index in attn_idx:
                    mine = global_heads(d, decode_tp, kv_heads)
                    if group.get("pattern") != "head-v1" or group.get("global_heads") != mine:
                        problems.append(f"decode rank {d} checked {check.get('req_id')}'s heads "
                                        f"{group.get('global_heads')} ({group.get('pattern')}), not its {mine}")
                    if not isinstance(group.get("bytes"), int) or group["bytes"] % len(mine):
                        problems.append(f"decode rank {d}: attention group {index} moved {group.get('bytes')} bytes "
                                        f"over {len(mine)} heads")
                        continue
                    per_head = group["bytes"] // len(mine)
                    if d == 0:
                        unique["attention"] += per_head * kv_heads
                    totals["attention"] += group["bytes"]
                    for p, heads in attention(d).items():
                        expected[(p, d)] = expected.get((p, d), 0) + per_head * len(heads)
                        attention_expected[(p, d)] = attention_expected.get((p, d), 0) + per_head * len(heads)
                    continue
                if index not in state_idx or group.get("pattern") != "state-v1":
                    problems.append(f"decode rank {d} checked group {index} as {group.get('pattern')}, "
                                    "not a state-v1 state group")
                    continue
                blocks, regions = group.get("blocks"), group.get("regions")
                if not (isinstance(blocks, int) and isinstance(regions, int) and blocks > 0 and regions > 0):
                    problems.append(f"decode rank {d}: state group {index} reports {blocks} blocks, {regions} regions")
                    continue
                if group.get("bytes") != blocks * regions * span_bytes:
                    problems.append(f"decode rank {d}: state group {index} moved {group.get('bytes')} bytes, not "
                                    f"{blocks} blocks x {regions} regions x {span_bytes} of its spans")
                transfer_only = group.get("transfer_only_bytes", 0)
                if transfer_only % (blocks * span_bytes):
                    problems.append(f"decode rank {d}: state group {index}'s transfer-only bytes {transfer_only} "
                                    f"are not whole regions of {span_bytes}")
                    continue
                only_regions = transfer_only // (blocks * span_bytes)
                logical_regions = regions - only_regions
                totals["state_transfer_only"] += transfer_only
                totals["state_logical"] += group["bytes"] - transfer_only
                if d == 0:
                    whole = sum(w for _, w in reference["components"]) * column + (
                        reference["ssm_heads"] * reference["ssm_head_bytes"])
                    unique["state_logical"] += blocks * logical_regions * whole
                    unique["state_transfer_only"] += blocks * only_regions * whole
                for sp in local:
                    width = (dict((n, w) for n, w in reference["components"])[sp[1]] if sp[0] == "conv"
                             else reference["ssm_heads"])
                    unit = column if sp[0] == "conv" else reference["ssm_head_bytes"]
                    for p, (lo, hi) in _state_overlap(d, decode_tp, prefill_tp, width).items():
                        for kind, regs in (("logical", logical_regions), ("transfer-only", only_regions)):
                            n = blocks * regs * (hi - lo) * unit
                            if n:
                                classes[(p, d, f"g{index}.{sp[0]}.{sp[1]}.{kind}")] = (
                                    classes.get((p, d, f"g{index}.{sp[0]}.{sp[1]}.{kind}"), 0) + n)
                                expected[(p, d)] = expected.get((p, d), 0) + n

    # Per-READ telemetry: exact bytes and lengths, then NIXL's three counts.
    logged: dict[tuple[int, int], int] = {}
    seen_classes: dict[tuple[int, int, str], int] = {}
    attention_seen: dict[tuple[int, int], int] = {}
    telemetry: dict[int, list[tuple]] = {d: [] for d in range(decode_tp)}
    descriptors, merged, unknown = [], [], 0
    for (_, rank), records in logs["decode"].items():
        for record in records:
            for entry in ((record.get("kv_transfers") or {}).get("load") or {}).get("each") or []:
                peer, plan = entry[5:7]
                if peer is None:
                    unknown += 1
                    continue
                at = f"prefill rank {peer} -> decode rank {rank}: "
                logged[(peer, rank)] = logged.get((peer, rank), 0) + int(entry[0])
                if not (isinstance(plan, list) and len(plan) > 4):
                    problems.append(f"{at}a READ without its selected bytes and lengths ({plan})")
                    continue
                selected, predicted, sel_bytes, table = plan[1], plan[2], plan[3], plan[4]
                measured = entry[3]
                if not (isinstance(table, dict) and isinstance(sel_bytes, int)):
                    problems.append(f"{at}a READ without its selected bytes and lengths ({plan[3:]})")
                    continue
                if int(entry[0]) != sel_bytes:
                    problems.append(f"{at}a READ of {entry[0]} bytes, its selected descriptors sum to {sel_bytes}")
                if sum(int(n) * c for n, c in table.items()) != sel_bytes or sum(table.values()) != selected:
                    problems.append(f"{at}a READ's length table {table} is not its {selected} descriptors of "
                                    f"{sel_bytes} bytes")
                count = lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0
                if not (count(selected) and count(measured) and count(predicted)):
                    problems.append(f"{at}a READ without counts: selected {selected}, predicted {predicted}, "
                                    f"measured {measured}")
                else:
                    descriptors.append(selected)
                    merged.append(measured)
                    if not predicted == measured <= selected:
                        problems.append(f"{at}a READ of {selected} selected descriptors predicted at {predicted} "
                                        f"and measured at {measured} (need predicted = measured <= selected)")
                    if descriptor_limit(run.server) is not None and selected > descriptor_limit(run.server):
                        problems.append(f"{at}a READ of {selected} selected descriptors, more than the "
                                        f"{NIXL_MAX_DESCRIPTORS} guard")
                telemetry[rank].append((peer, int(entry[0]), selected))
    if unknown:
        problems.append(f"{unknown} transfer(s) logged without their peer rank")
    for d in range(decode_tp):
        posted = sorted((r.get("rank"), r.get("bytes"), r.get("descriptors")) for c in checks[d]
                        for r in (c.get("coverage") or {}).get("reads") or [])
        for c in checks[d]:
            for r in (c.get("coverage") or {}).get("reads") or []:
                for name, (_, n) in (r.get("classes") or {}).items():
                    if ".attention." in name:
                        attention_seen[(r.get("rank"), d)] = attention_seen.get((r.get("rank"), d), 0) + n
                        continue
                    key = (r.get("rank"), d, name)
                    seen_classes[key] = seen_classes.get(key, 0) + n
        if posted != sorted(telemetry[d]):
            problems.append(f"decode rank {d}'s checked READs {posted[:3]}... are not its logged READs "
                            f"{sorted(telemetry[d])[:3]}... (producer, bytes, selected descriptors)")
    for key in sorted(set(classes) | set(seen_classes)):
        if classes.get(key, 0) != seen_classes.get(key, 0):
            problems.append(f"producer {key[0]} -> decode rank {key[1]}: {key[2]} carried {seen_classes.get(key, 0)} "
                            f"bytes, not the {classes.get(key, 0)} its owned range gives")
    for pair in sorted(set(attention_expected) | set(attention_seen)):
        if attention_expected.get(pair, 0) != attention_seen.get(pair, 0):
            problems.append(f"prefill rank {pair[0]} -> decode rank {pair[1]}: attention carried "
                            f"{attention_seen.get(pair, 0)} bytes, not the {attention_expected.get(pair, 0)} "
                            "of the heads it holds for it")
    for pair in sorted(set(expected) | set(logged)):
        if logged.get(pair, 0) != expected.get(pair, 0):
            problems.append(f"prefill rank {pair[0]} -> decode rank {pair[1]} moved {logged.get(pair, 0)} bytes, "
                            f"not the {expected.get(pair, 0)} its owned heads and state ranges give")
    for p in sorted(unwritten):
        problems.append(f"prefill rank {p}'s TENT metrics lack tent_write_bytes_total")
    received = {d: sum(n for (p, dd), n in logged.items() if dd == d) for d in range(decode_tp)}
    sent = {p: sum(n for (pp, d), n in logged.items() if pp == p) for p in range(prefill_tp)}
    payload = sum(logged.values())
    moved = sum(summary["transfers"]["bytes"] for summary in ranks["decode"].values())
    if moved != payload:
        problems.append(f"the connector moved {moved} bytes, not the {payload} its transfers' peers account for")
    for p in range(prefill_tp):
        produced = ranks["prefill"].get((0, p), {}).get("requests") or set()
        if run.kv_check and len(produced) != run.workload.num_prompts:
            problems.append(f"prefill rank {p} produced {len(produced)} request(s), not {run.workload.num_prompts}")
    port_evidence = {}
    if ports is not None:
        network = ports.get(run.prefill.kv_network) or {}
        by_rank = {(described["node"], described["slot"]): described for described in network.values()}
        for role, node, carried, direction in (("prefill", run.prefill.node, sent, "rx"),
                                               ("decode", run.decode.node, received, "tx")):
            for rank, payload_bytes in carried.items():
                port = by_rank.get((node, rank))
                if port is None:
                    problems.append(f"{role} rank {rank}'s router port was not measured")
                    continue
                measured = port[direction]
                port_evidence[f"{role}{rank}"] = {"rx": port["rx"], "tx": port["tx"], "payload": payload_bytes}
                if measured < payload_bytes or measured > payload_bytes * 1.05 + (1 << 20):
                    problems.append(f"{role} rank {rank}'s router port {direction.upper()} {measured} bytes, not its "
                                    f"{payload_bytes} of KV and little more")
    evidence.update({
        "pairs": [{"prefill": p, "decode": d, "bytes": logged.get((p, d), 0), "expected": expected.get((p, d), 0)}
                  for p in range(prefill_tp) for d in range(decode_tp) if (p, d) in expected or (p, d) in logged],
        "byte_classes": totals, "unique_bytes_by_class": unique, "payload_bytes": payload,
        "max_descriptors": max(descriptors, default=None), "max_merged_descriptors": max(merged, default=None),
        "state_classes": {f"{p}->{d} {name}": n for (p, d, name), n in sorted(classes.items())},
        "ports": port_evidence or None, "received": received, "problems": problems,
    })
    return evidence



def _hp_tcp_writes(records: list[dict], limit: int | None) -> dict:
    """A Mooncake prefill rank's connector writes (``send``) against hp_tcp's
    task limit, reported, never judged: TENT may merge a write's descriptors
    into fewer tasks, and how many of another write's tasks are still
    outstanding when one is submitted is not logged. Each write spans
    [end - seconds, end]; a failed one [start, its record's time] (its end is
    not logged; the record reporting it is written after it). The
    overlapping-call bound is the most descriptors whose spans overlap: an
    upper bound on what was in flight, complete only if every write's span
    is known."""
    spans: list[tuple[float, float, int]] = []
    largest: list[int] = []
    complete = True
    written = failed = 0
    for record in records:
        send = (record.get("kv_transfers") or {}).get("send") or {}
        each = send.get("each") or []
        written += int(send.get("count") or 0)
        complete &= len(each) == int(send.get("count") or 0)
        for entry in each:
            descriptors = entry[3]
            if descriptors is None:
                complete = False
                continue
            largest.append(descriptors)
            if entry[1] is None or entry[4] is None:
                complete = False
                continue
            spans.append((entry[4] - entry[1], entry[4], descriptors))
        for entry in send.get("failed_each") or []:
            failed += 1
            descriptors, start = entry[1:3]
            if descriptors is None:
                complete = False
                continue
            largest.append(descriptors)
            end = record.get("t")
            if start is None or end is None or end < start:
                complete = False
                continue
            spans.append((start, end, descriptors))
        # Failures counted but not each listed have no span.
        complete &= len(send.get("failed_each") or []) >= int(send.get("failed") or 0)
        failed += max(0, int(send.get("failed") or 0) - len(send.get("failed_each") or []))
    in_flight = peak = 0
    for _, change in sorted([(a, n) for a, _, n in spans] + [(b, -n) for _, b, n in spans],
                            key=lambda event: (event[0], event[1])):
        in_flight += change
        peak = max(peak, in_flight)
    notes = []
    biggest = max(largest, default=None)
    if limit is None:
        notes.append("hp_tcp's task limit is unknown (server.env replaces TENT's configuration)")
    else:
        if biggest is not None and biggest > limit:
            notes.append(f"a write of {biggest} connector descriptors exceeds the task limit {limit} (TENT may "
                         "merge them: its tasks are not logged)")
        if peak > limit:
            notes.append(f"writes of {peak} connector descriptors overlapped, above the task limit {limit} (an "
                         "upper bound: their tasks complete before their calls return)")
    return {"task_limit": limit, "connector_writes": written, "failed_connector_writes": failed,
            "connector_descriptors": sum(largest), "max_connector_descriptors_per_write": biggest,
            "overlapping_call_descriptor_bound": peak, "bound_complete": complete, "notes": notes}



def _admissions(records: list[dict]) -> dict[str, list[dict]]:
    """Each request's admissions in a rank's step log, in order: a new or
    resumed request's tokens computed when admitted (``admitted``), its
    blocks then by group (``allocated_groups`` from place 0; the null block,
    0, holding a window's clipped places) and, when a pull was posted for it
    before, that pull's (first place, blocks) by group."""
    admissions: dict[str, list[dict]] = {}
    posted: dict[str, dict[int, tuple[int, list[int]]]] = {}
    for record in records:
        events = record.get("kv_events") or {}
        by_group = events.get("allocated_groups") or {}
        for request in events.get("reads_posted") or []:
            posted[request] = {entry["group"]: (entry["first"], list(entry["blocks"]))
                               for entry in by_group.get(request, [])}
        for request, computed in (events.get("admitted") or {}).items():
            admissions.setdefault(request, []).append({
                "computed": computed,
                "table": {entry["group"]: list(entry["blocks"]) for entry in by_group.get(request, [])
                          if entry["first"] == 0},
                "pull": posted.pop(request, None)})
    return admissions



def _hit_problems(layout: dict, hits: list[dict], admissions: dict[str, list[dict]]) -> tuple[list[str], list[dict]]:
    """A rank's prefix hit checks against its requests' admissions, and the
    checks that agree with them. Each admission implies a hit of each
    group's places before its pull's first (decode, a pull posted), else
    before its computed tokens' full blocks: its blocks there, null ones
    (a window's clipped head) left out. Each admission with such blocks has
    one hit check, in order, of exactly those places and blocks, its bytes
    the layout's for them; no request has any other. A pull's blocks must be
    the request's at their places when admitted, its first within the
    admitted computed tokens."""
    groups = layout.get("groups")
    if not groups:
        return ["the layout lists no KV cache groups to check prefix hits by"], []
    problems: list[str] = []
    expected: dict[str, list[dict[int, list[tuple[int, int]]]]] = {}
    for request, adms in admissions.items():
        for admission in adms:
            pull, table = admission["pull"], admission["table"]
            for group, (first, blocks) in (pull or {}).items():
                size = groups[group]["block_size"] if 0 <= group < len(groups) else None
                if table.get(group, [])[first:first + len(blocks)] != blocks:
                    problems.append(f"{request}'s pull of group {group} ({len(blocks)} block(s) from {first}) is "
                                    "not its blocks there when admitted")
                elif size is None or first * size > admission["computed"]:
                    problems.append(f"{request}'s pull of group {group} starts at block {first}, past the "
                                    f"{admission['computed']} tokens it was admitted with")
            want = {}
            for group, spec in enumerate(groups):
                count = (pull.get(group, (0, []))[0] if pull is not None
                         else admission["computed"] // spec["block_size"])
                row = table.get(group, [])
                places = [(place, row[place]) for place in range(min(count, len(row))) if row[place]]
                if places:
                    want[group] = places
            if want:
                expected.setdefault(request, []).append(want)
    logged: dict[str, list[dict]] = {}
    for check in hits:
        if "groups" in check:  # a hit outside the lifecycle has none, reported apart
            logged.setdefault(check.get("req_id"), []).append(check)
    valid = []
    for request in sorted(set(expected) | set(logged), key=str):
        want, got = expected.get(request, []), logged.get(request, [])
        if len(want) != len(got):
            problems.append(f"{request} logged {len(got)} prefix hit check(s), not the {len(want)} its "
                            f"admissions imply ({[sum(map(len, w.values())) for w in want]} block(s))")
        for places, check in zip(want, got):
            found = []
            seen = {}
            for g in check["groups"]:
                if any(key not in g for key in ("group", "logical_blocks", "block_ids", "blocks", "bytes")):
                    found.append(f"{request}'s hit check of group {g.get('group')} lacks its places, blocks or "
                                 "bytes")
                    continue
                spec = groups[g["group"]] if 0 <= g["group"] < len(groups) else None
                seen[g["group"]] = list(zip(g["logical_blocks"], g["block_ids"]))
                per_block = None if spec is None else (
                    spec["transfer"]["bytes_per_block"] if "transfer" in spec
                    else spec.get("checked_bytes_per_block") or spec["layers"] * spec["block_bytes"])
                if (len(g["logical_blocks"]), len(g["block_ids"])) != (g["blocks"], g["blocks"]) or (
                        per_block is None or g["bytes"] != g["blocks"] * per_block):
                    found.append(f"{request}'s hit check of group {g['group']} ({g['blocks']} block(s), "
                                 f"{g['bytes']} bytes) disagrees with the layout ({per_block} bytes a block)")
            if seen != places:
                found.append(f"{request}'s hit check covers {_places(seen)}, not its admitted hit "
                             f"{_places(places)}")
            if (check.get("blocks"), check.get("bytes")) != (sum(g.get("blocks", 0) for g in check["groups"]),
                                                             sum(g.get("bytes", 0) for g in check["groups"])):
                found.append(f"{request}'s hit check is not its groups' sum")
            if not (check.get("blocks") and check.get("bytes")):
                found.append(f"{request}'s hit check covers no bytes")
            problems.extend(found)
            if not found and check.get("ok") and not check.get("mismatched_bytes"):
                valid.append(check)
    return _capped(problems), valid



def _places(places: dict[int, list[tuple[int, int]]]) -> str:
    """Hit places by group, briefly: each group's count and first and last
    (place: block)."""
    return "; ".join(f"group {group}: {len(p)} block(s)" + (f" {p[0][0]}:{p[0][1]}..{p[-1][0]}:{p[-1][1]}" if p else "")
                     for group, p in sorted(places.items())) or "nothing"



def _request_steps(records: list[dict]) -> dict[str, list[tuple[int, int, int, int]]]:
    """Each request's steps in a step log: (computed, scheduled, prompt,
    speculative) tokens, in order; speculative, the drafts the step
    verifies among its scheduled tokens."""
    steps: dict[str, list[tuple[int, int, int, int]]] = {}
    for record in records:
        for request in (record.get("step") or {}).get("requests", []):
            steps.setdefault(request["req_id"], []).append(
                (request["num_computed_tokens"], request["num_scheduled_tokens"], request["num_prompt_tokens"],
                 request.get("num_spec_tokens") or 0))
    return steps



def _handoff_from_sources(joint_steps: dict, decode_rank: int, sources: list[int], consumed: set[str]) -> dict:
    """Validate the handoff on every producer that supplies this decode rank.

    TP ranks need not match: scatter shares a producer, while gather needs
    each producer's steps. Keep the request count per decode rank.
    """
    results = [_handoff_problems(joint_steps[("prefill", source)], joint_steps[("decode", decode_rank)], consumed)
               for source in sources]
    evidence = results[0]["evidence"]
    return {"evidence": evidence, "problems": [
        (f"prefill rank {source}: " if sources != [decode_rank] else "") + problem
        for source, result in zip(sources, results)
        for problem in result["problems"]]}



def _handoff_problems(prefill: dict[str, list[tuple[int, int, int, int]]],
                      decode: dict[str, list[tuple[int, int, int, int]]],
                      consumed: set[str]) -> dict:
    """A hybrid model's handoff, from both sides' steps: prefill computes
    the prompt but its last token (Mooncake truncates its N tokens to
    N - 1), decode resumes at N - 1 and computes the last one. Every
    request decode consumed (`consumed`, by its own request ids: its byte
    checks, or its allocations) needs its steps. Requests whose steps show
    one prompt token, which no side truncates, are not checked."""
    produced = {_base_request_id(request): steps for request, steps in prefill.items()}
    problems, prompts = [], []
    for request in sorted(consumed):
        steps = decode.get(request)
        if steps is None:
            problems.append(f"{request} has no decode steps")
            continue
        computed, scheduled, prompt, drafts = steps[0]
        if prompt <= 1:
            continue
        prompts.append(prompt)
        # The last prompt token, and with speculative decoding the drafts
        # the same step verifies.
        if (computed, scheduled) != (prompt - 1, 1 + drafts):
            problems.append(f"{request} resumed at {computed} tokens with {scheduled} scheduled, not at "
                            f"{prompt - 1} with {1 + drafts}")
        steps = produced.get(_base_request_id(request))
        if steps is None:
            problems.append(f"{request} has no prefill steps")
            continue
        computed, scheduled, prefilled, _ = steps[-1]
        if prefilled != prompt - 1 or computed + scheduled != prompt - 1:
            problems.append(f"{request}'s prefill took a {prefilled}-token prompt to {computed + scheduled} "
                            f"tokens, not its {prompt} tokens but the last")
    return {"problems": _capped(problems), "evidence": {
        "requests": len(prompts), "prompt_tokens": [min(prompts), max(prompts)] if prompts else None,
        "prefilled": "N - 1", "decode_resumes_at": "N - 1"}}



def _capped(problems: list[str], keep: int = 3) -> list[str]:
    """At most `keep` problems of one kind, then their count."""
    return problems if len(problems) <= keep else problems[:keep] + [
        f"and {len(problems) - keep} more such problems"]



def _group_evidence(layout: dict, checks: list[dict], records: list[dict]) -> list[dict]:
    """Each KV cache group of a rank: its layout, what decode checked of it
    (blocks, bytes, pulls clipped to a window and the furthest first block)
    and its allocations by place (`allocated_groups`): places held by the
    null block (a window's head), blocks a request got at more than one
    place (reused), and blocks given at more than one place of any requests
    (reallocated)."""
    given: list[set[tuple[str, int, int]]] = [set() for _ in layout["groups"]]
    for record in records:
        for request, entries in ((record.get("kv_events") or {}).get("allocated_groups") or {}).items():
            for entry in entries:
                given[entry["group"]].update(
                    (request, entry["first"] + offset, block) for offset, block in enumerate(entry["blocks"]))
    evidence = []
    for index, spec in enumerate(layout["groups"]):
        nested = [g for check in checks for g in check.get("groups", []) if g["group"] == index]
        firsts = [g["first_logical_block"] for g in nested]
        places: dict[int, set[tuple[str, int]]] = {}
        for request, place, block in given[index]:
            if block:
                places.setdefault(block, set()).add((request, place))
        evidence.append({
            **spec,
            "checked_blocks": sum(g["blocks"] for g in nested),
            "checked_bytes": sum(g["bytes"] for g in nested),
            "clipped_pulls": sum(first > 0 for first in firsts),
            "first_logical_block_max": max(firsts, default=None),
            "null_places": len({(request, place) for request, place, block in given[index] if not block}),
            "reused_blocks": sum(len(held) > len({request for request, _ in held}) for held in places.values()),
            "reallocated_blocks": sum(len(held) > 1 for held in places.values()),
        })
    return evidence



def _base_request_id(request_id: str) -> str:
    """The id the proxy gives a request on both instances: vLLM appends its
    own random suffix to each (``cmpl-<uuid>-0-<suffix>``)."""
    return request_id.rsplit("-", 1)[0]



def _records(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]

