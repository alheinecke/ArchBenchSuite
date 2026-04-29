#!/usr/bin/env python3
# Copyright (c) 2026, Alexander Heinecke
# gen_agentic_schedule.py — Generate per-thread schedule files for AgenticCPUBench
# that model an agentic-AI CPU workload mix.
#
# Output files are named <prefix>_tid<N>.sched and contain one line per round:
#     <bench_index> <reps_multiplier>
# matching the format consumed by `--replay-schedule <prefix>`.
#
# Replay with:
#   ./AgenticCPUBench_<ISA>.exe all <rounds> --replay-schedule <prefix>
#
# Workload model
# --------------
# Bench-mix proportions per thread are chosen to approximate an agentic AI
# worker thread:
#
#   sleep      — I/O waits, LLM API round-trips, MCP / shell calls, lock waits
#   intipc     — integer-IPC heavy work: parsers, compilers, regex, hashing,
#                serialisation, JIT codegen
#   qs         — branch-heavy: DBMS query exec, search/ranking, FS ops,
#                schedulers, GC
#   cachebwl2  — small/hot data processing: tokenisation, hot-loop interp,
#                short table scans
#   cachebwl3  — medium data processing: column scans, log filtering,
#                JSON/CSV chunks
#   triad      — large streaming I/O from a single core: file copy/memcpy,
#                network buffer drains, RPC payload (de)serialisation
#   latency    — pointer-chasing / random access: graph algorithms, SpMV,
#                embedding/KV lookups, hash-table probes
#   xsmm       — SIMD compute: ML inference (GEMM), DSP/FFT, dense linalg
#
# To make threads heterogeneous (different roles concurrently active) we
# additionally bias each thread toward a "role" that perturbs the base mix.

import argparse
import os
import random
import sys

# Bench index order MUST match the order in which AgenticCPUBench.cpp pushes
# benchmarks into `benches` when `all` is selected. See
# AgenticCPUBench.cpp:main → run_triad / run_cachebwl2 / ... blocks.
BENCH_INDEX = {
    "triad":     0,
    "cachebwl2": 1,
    "cachebwl3": 2,
    "xsmm":      3,
    "qs":        4,
    "intipc":    5,
    "latency":   6,
    "sleep":     7,
}

# Base bench mix (probabilities, summing to 1.0). Reflects the typical
# agentic worker: lots of waiting on external calls, a healthy share of
# parsing/compilation and search, modest streaming and compute bursts.
BASE_MIX = {
    "sleep":     0.32,   # blocking I/O / LLM / MCP / tool calls
    "intipc":    0.16,   # parsers, compilers, regex, hashing
    "qs":        0.13,   # search, ranking, DBMS, schedulers
    "cachebwl2": 0.10,   # small data transforms
    "cachebwl3": 0.10,   # medium scans / column ops
    "triad":     0.08,   # file I/O, RPC payloads, memcpy
    "latency":   0.06,   # graph / KV / embedding lookups
    "xsmm":      0.05,   # GEMM / inference bursts
}

# Upper bound (inclusive) for the per-round random reps multiplier. MUST
# match RND_REPS in AgenticCPUBench.cpp.
RND_REPS = 30

# Per-role overlays. Each role tweaks BASE_MIX (multiplicative weights) to
# approximate a specialised agent worker. The 8 default roles below cover
# the typical population in a multi-agent system.
ROLE_WEIGHTS = {
    # Orchestrator: top-level planner that decomposes goals, dispatches
    # sub-tasks to worker agents and waits on their results. Mostly waiting
    # on LLM and worker round-trips with light dispatch bookkeeping.
    #   - LLM planning round-trips and worker join waits    -> sleep (dominant)
    #   - parsing/serialising plans, tasks, status updates  -> intipc
    #   - small string ops on prompts / task descriptors    -> cachebwl2
    #   - prioritising / re-ordering pending tasks          -> qs
    #   - task-graph and worker-state lookups               -> latency
    #   - light scans of conversation / status logs         -> cachebwl3
    #   - rare bulk transcript dump                         -> triad (light)
    #   - essentially no dense SIMD compute                 -> xsmm (rare)
    "orchestrator": {"sleep": 1.6, "intipc": 1.1, "qs": 0.8, "xsmm": 0.4,
                     "triad": 0.7, "latency": 0.6},
    # LLM-driven coding agent (e.g. Copilot/Claude-style tool-using agent):
    # mirrors what an LLM agent does while editing/debugging a codebase:
    #   - tool-call / model round-trip and subagent waits   -> sleep (dominant)
    #   - tokenisation, JSON/protobuf serialisation, regex,
    #     hashing of tool I/O                               -> intipc
    #   - sorting/ranking candidate matches and edits       -> qs
    #   - small string ops on prompts / snippets            -> cachebwl2
    #   - scanning fetched source files / build logs        -> cachebwl3
    #   - hopping across the symbol/xref graph (defs,
    #     usages, includes, dedup of seen files)            -> latency
    #   - rare bulk file copy / large diff write-out        -> triad (light)
    #   - essentially no dense SIMD compute                 -> xsmm (rare)
    "llm_coder":    {"sleep": 2.0, "intipc": 1.5, "qs": 1.3,
                     "cachebwl2": 1.2, "cachebwl3": 1.5, "latency": 1.4,
                     "triad": 0.6, "xsmm": 0.3},
    # Researcher: web searches, fetching pages / PDFs, following citation and
    # paper links. Mirrors what an LLM agent does when chasing references:
    #   - heavy HTTP / search-API waits                     -> sleep (dominant)
    #   - scanning fetched HTML/PDF/markdown text           -> cachebwl3
    #   - HTML/PDF/JSON parsing, tokenisation, dedup hash   -> intipc
    #   - hopping across the link/citation graph (URL set,
    #     bibliography lookups, dedup of seen URLs)         -> latency
    #   - ranking / re-ranking candidate results            -> qs
    #   - small string ops on snippets / titles             -> cachebwl2
    #   - occasional bulk download of a large PDF           -> triad (modest)
    #   - essentially no dense SIMD compute                 -> xsmm (rare)
    "researcher":   {"sleep": 2.4, "cachebwl3": 1.8, "intipc": 1.4,
                     "latency": 1.8, "qs": 1.2, "cachebwl2": 1.1,
                     "triad": 0.8, "xsmm": 0.2},
    # Personal assistant / chat agent: handles conversational requests,
    # checks calendars, books and reschedules appointments, sends
    # confirmations. Heavy on small interactive turns and short calendar /
    # contact lookups; very little bulk compute.
    #   - LLM turn waits, calendar/email/CRM API calls, user idle gaps
    #     between messages                                  -> sleep (dominant)
    #   - JSON/iCal/protobuf (de)serialisation, request
    #     validation, hashing of message/event ids          -> intipc
    #   - small string ops on chat tokens, names, titles    -> cachebwl2
    #   - scanning conversation/calendar context windows    -> cachebwl3
    #   - sorting candidate slots, ranking contacts /
    #     suggested times                                   -> qs
    #   - calendar / contact / availability index lookups
    #     (KV-store probes, free/busy graph)                -> latency
    #   - rare bulk export (e.g. itinerary PDF, ICS file)   -> triad (light)
    #   - essentially no dense SIMD compute                 -> xsmm (rare)
    "assistant":    {"sleep": 2.6, "intipc": 1.2, "cachebwl2": 1.4,
                     "cachebwl3": 1.0, "qs": 1.1, "latency": 1.5,
                     "triad": 0.5, "xsmm": 0.2},
    # RAG / retrieval worker: serves queries by embedding-table lookups,
    # vector-DB / inverted-index probes, and small inference passes for
    # re-ranking and answer composition.
    #   - vector-DB / KV / embedding-table probes,
    #     inverted-index hops                               -> latency (dominant)
    #   - scanning candidate document/passage chunks        -> cachebwl3
    #   - top-k / re-rank sorting of candidates             -> qs
    #   - re-rank / cross-encoder / small inference         -> xsmm
    #   - JSON request/response, tokenisation, hashing      -> intipc
    #   - small string ops on snippets / titles             -> cachebwl2
    #   - DB / network round-trips, paging waits            -> sleep (modest)
    #   - occasional bulk passage / index-shard transfer    -> triad (modest)
    "rag":          {"latency": 2.5, "cachebwl3": 1.5, "qs": 1.3, "xsmm": 1.4,
                     "sleep": 0.8, "triad": 1.0},
    # Tool runner: executes external tools / shell commands / MCP servers
    # and shuttles their I/O back. Dominated by waiting on the spawned
    # process and moving its output around.
    #   - subprocess / RPC / MCP-call waits                 -> sleep (dominant)
    #   - draining stdout/stderr / pipe buffers, large
    #     file copy of tool artefacts                       -> triad
    #   - small string ops on argv / env / short tool I/O   -> cachebwl2
    #   - light parsing of tool output (JSON, text)         -> intipc
    #   - simple sorting / dedup of result lists            -> qs
    #   - occasional symbol/path lookups                    -> latency
    #   - essentially no dense SIMD compute                 -> xsmm (rare)
    "tool":         {"sleep": 2.2, "triad": 1.3, "cachebwl2": 1.2,
                     "intipc": 0.8, "xsmm": 0.3, "latency": 0.6},
    # Data wrangler / ETL: reads tables and logs, projects/filters/joins,
    # sorts and writes results back. Streaming and scan heavy with modest
    # compute and few external waits.
    #   - column / row scans, log filtering, JSON/CSV chunks -> cachebwl3 (dominant)
    #   - tokenisation, small projections, hot transforms    -> cachebwl2
    #   - sort / merge / hash-join build phases              -> qs
    #   - file/network ingest and result writeback           -> triad
    #   - parsing, schema validation, hashing of keys        -> intipc
    #   - I/O completion / disk waits                        -> sleep (modest)
    #   - hash-table probes for joins / dedup                -> latency
    #   - light vectorised aggregations                      -> xsmm (small)
    "data":         {"cachebwl3": 1.8, "cachebwl2": 1.5, "qs": 1.4,
                     "triad": 1.5, "sleep": 0.7, "xsmm": 0.5},
    # Inference worker: runs ML model kernels (GEMM/attention/conv) for
    # local model inference or re-ranking. Compute-bound with modest data
    # movement around the kernels.
    #   - dense SIMD compute (GEMM / attention / conv)      -> xsmm (dominant)
    #   - activation / KV-cache scans between layers        -> cachebwl3
    #   - weight / activation streaming                     -> triad
    #   - tokenisation, request (de)serialisation           -> intipc
    #   - sampling / top-k / argmax over logits             -> qs
    #   - KV-cache / embedding-table lookups                -> latency
    #   - batch-queue / scheduler waits                     -> sleep (modest)
    "inference":    {"xsmm": 4.0, "cachebwl3": 1.2, "triad": 1.1,
                     "sleep": 0.5, "intipc": 0.7, "qs": 0.6, "latency": 0.8},
    # Graph analytics: BFS / PageRank / connected-components / SpMV style
    # workloads dominated by irregular pointer chasing.
    #   - random-access pointer chasing, frontier hops      -> latency (dominant)
    #   - sorting / partitioning of frontiers and edges     -> qs
    #   - scans of CSR / adjacency arrays                   -> cachebwl3
    #   - bitset / id ops on small per-vertex state         -> cachebwl2
    #   - ranking / reduction across active vertices        -> intipc
    #   - small SIMD reductions (dot-products on chunks)    -> xsmm (small)
    #   - shard / partition swap to/from disk               -> triad
    #   - barrier / superstep waits                         -> sleep (modest)
    "graph":        {"latency": 3.0, "qs": 1.5, "cachebwl3": 1.2,
                     "sleep": 0.6, "xsmm": 0.5, "intipc": 0.8},
    # Idle / observer: a mostly-quiet thread that pings health checks and
    # waits on events. Models the always-present "slack" cores in a
    # multi-agent system.
    #   - long event/queue/condvar waits, heartbeat sleeps  -> sleep (dominant)
    #   - tiny string ops on heartbeat / status payloads    -> cachebwl2
    #   - light parsing / hashing of status messages        -> intipc
    #   - small sort / dedup of pending events              -> qs
    #   - occasional state-table lookups                    -> latency
    #   - rare log rotation / dump                          -> triad (light)
    #   - essentially no dense SIMD compute                 -> xsmm (rare)
    "idle":         {"sleep": 3.0, "cachebwl2": 0.8, "intipc": 0.6,
                     "qs": 0.5, "xsmm": 0.2, "triad": 0.5, "latency": 0.5},
}


def normalise(weights):
    s = sum(weights.values())
    return {k: v / s for k, v in weights.items()}


def role_mix(role):
    """Combine BASE_MIX with the role overlay and return a normalised dict."""
    overlay = ROLE_WEIGHTS.get(role, {})
    mix = {}
    for name, p in BASE_MIX.items():
        mix[name] = p * overlay.get(name, 1.0)
    return normalise(mix)


def draw_bench(mix, rng):
    """Draw a benchmark name from a normalised mix dict."""
    r = rng.random()
    acc = 0.0
    for name, p in mix.items():
        acc += p
        if r <= acc:
            return name
    return next(reversed(mix))


def draw_reps(bench_name, role, rng):
    """Draw a reps multiplier in [1, RND_REPS] biased per benchmark.

    Agentic workloads have very heterogeneous burst lengths:
      - sleep: mostly short waits, occasional long blocking calls.
      - xsmm:  a few short bursts, occasional long inference batches.
      - intipc/qs/cachebw/triad/latency: medium, somewhat dispersed.
    """
    # Per-bench shape parameters (alpha, beta) for a Beta distribution
    # over [0,1] which we then map to [1, RND_REPS].
    shape = {
        # Small-mean, long tail (most are short, occasional long bursts):
        "sleep":     (1.5, 6.0),
        "tool":      (1.5, 6.0),
        # Compute / inference bursts: occasionally long.
        "xsmm":      (1.8, 3.5),
        # Streaming I/O: variable but often medium-sized.
        "triad":     (2.0, 3.0),
        # Random access / graph traversal: medium with spread.
        "latency":   (2.0, 3.0),
        # Mid-spread for parsing / search / scans.
        "intipc":    (2.2, 2.8),
        "qs":        (2.2, 2.8),
        "cachebwl2": (2.2, 2.8),
        "cachebwl3": (2.2, 2.8),
    }
    a, b = shape.get(bench_name, (2.0, 3.0))
    # Inference role pushes xsmm bursts even longer.
    if role == "inference" and bench_name == "xsmm":
        a, b = 3.5, 1.8
    # Idle role pushes sleep waits even longer.
    if role == "idle" and bench_name == "sleep":
        a, b = 1.2, 2.5
    # LLM coder: many medium-length tool/model round-trips with occasional
    # very long subagent / build calls.
    if role == "llm_coder" and bench_name == "sleep":
        a, b = 1.8, 4.0
    # Researcher: HTTP fetches and search-API calls — medium waits with a
    # heavy tail for slow servers / large PDF downloads / PDF rendering.
    if role == "researcher" and bench_name == "sleep":
        a, b = 1.6, 4.5
    # Researcher: scans of fetched pages can be quite long for big PDFs.
    if role == "researcher" and bench_name == "cachebwl3":
        a, b = 2.5, 2.5
    # Personal assistant: chat turns are mostly short LLM round-trips with
    # occasional long user-idle gaps (waiting for a reply / confirmation).
    if role == "assistant" and bench_name == "sleep":
        a, b = 1.4, 5.0

    u = rng.betavariate(a, b)             # u in (0,1)
    k = 1 + int(round(u * (RND_REPS - 1)))
    return max(1, min(RND_REPS, k))


def gen_thread_schedule(role, rounds, rng):
    mix = role_mix(role)
    schedule = []
    for _ in range(rounds):
        bench = draw_bench(mix, rng)
        k = draw_reps(bench, role, rng)
        schedule.append((BENCH_INDEX[bench], k, bench))
    return schedule


def write_schedule(prefix, tid, schedule):
    fname = f"{prefix}_tid{tid}.sched"
    with open(fname, "w") as f:
        for idx, k, _name in schedule:
            f.write(f"{idx} {k}\n")
    return fname


def print_summary(role, schedule):
    counts = {}
    reps_sum = {}
    for _idx, k, name in schedule:
        counts[name] = counts.get(name, 0) + 1
        reps_sum[name] = reps_sum.get(name, 0) + k
    total = len(schedule)
    print(f"  role={role:<12s}  rounds={total}")
    for name in sorted(counts.keys()):
        c = counts[name]
        avg_k = reps_sum[name] / c
        print(f"    {name:<10s} count={c:>4d}  frac={c/total:>6.1%}  "
              f"mean_k={avg_k:>5.2f}  total_k={reps_sum[name]:>5d}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate AgenticCPUBench schedule files modelling "
                    "an agentic-AI CPU workload across multiple cores."
    )
    p.add_argument("--prefix", default="agentic",
                   help="Output filename prefix (default: agentic). "
                        "Files written: <prefix>_tid<N>.sched")
    p.add_argument("--threads", type=int, default=8,
                   help="Number of threads / schedule files (default: 8).")
    p.add_argument("--rounds", type=int, default=200,
                   help="Number of rounds per thread (default: 200).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility (default: 42).")
    p.add_argument("--roles", nargs="*", default=None,
                   help="Optional explicit list of roles, one per thread. "
                        f"Available: {sorted(ROLE_WEIGHTS.keys())}. "
                        "If omitted, a role is drawn uniformly at random "
                        "per thread from the available role catalog.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-thread mix summary.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.threads <= 0 or args.rounds <= 0:
        print("error: --threads and --rounds must be positive.",
              file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)

    if args.roles:
        unknown = [r for r in args.roles if r not in ROLE_WEIGHTS]
        if unknown:
            print(f"error: unknown role(s): {unknown}.\n"
                  f"available: {sorted(ROLE_WEIGHTS.keys())}",
                  file=sys.stderr)
            sys.exit(1)
        if len(args.roles) != args.threads:
            print(f"error: --roles has {len(args.roles)} entries but "
                  f"--threads={args.threads}.", file=sys.stderr)
            sys.exit(1)
        roles = list(args.roles)
    else:
        # Pick a random role per thread from the available role catalog.
        available = sorted(ROLE_WEIGHTS.keys())
        roles = [rng.choice(available) for _ in range(args.threads)]

    if not args.quiet:
        print(f"Generating {args.threads} schedule(s), {args.rounds} rounds each,"
              f" RND_REPS={RND_REPS}, seed={args.seed}")
        print(f"Output files: {args.prefix}_tid0..{args.threads - 1}.sched")
        print()

    out_files = []
    for tid in range(args.threads):
        # Each thread gets its own RNG stream derived from the master seed
        # so individual thread schedules are reproducible independently.
        thread_rng = random.Random(rng.random())
        sched = gen_thread_schedule(roles[tid], args.rounds, thread_rng)
        fname = write_schedule(args.prefix, tid, sched)
        out_files.append(fname)
        if not args.quiet:
            print_summary(roles[tid], sched)

    if not args.quiet:
        print()
        print(f"Wrote {len(out_files)} schedule file(s).")
        print()
        print("Replay with:")
        print(f"  OMP_NUM_THREADS={args.threads} \\")
        print(f"  LD_LIBRARY_PATH=./libxsmm/lib \\")
        print(f"  ./AgenticCPUBench_<ISA>.exe all {args.rounds} "
              f"--replay-schedule {args.prefix}")


if __name__ == "__main__":
    main()
