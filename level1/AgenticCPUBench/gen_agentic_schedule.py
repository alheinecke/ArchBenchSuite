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

# Per-role overlays. Each role tweaks BASE_MIX (multiplicative weights) to
# approximate a specialised agent worker. The 8 default roles below cover
# the typical population in a multi-agent system.
ROLE_WEIGHTS = {
    # Orchestrator: lots of LLM round-trips, light parsing, little compute.
    "orchestrator": {"sleep": 1.6, "intipc": 1.1, "qs": 0.8, "xsmm": 0.4,
                     "triad": 0.7, "latency": 0.6},
    # Coder agent: compilation, file scans, sort/search.
    "coder":        {"intipc": 2.0, "qs": 1.4, "cachebwl3": 1.3, "triad": 1.2,
                     "sleep": 0.7, "xsmm": 0.4},
    # RAG / retrieval: embedding lookups, DB scans, some inference.
    "rag":          {"latency": 2.5, "cachebwl3": 1.5, "qs": 1.3, "xsmm": 1.4,
                     "sleep": 0.8, "triad": 1.0},
    # Tool runner: dominated by external tool invocations and light I/O.
    "tool":         {"sleep": 2.2, "triad": 1.3, "cachebwl2": 1.2,
                     "intipc": 0.8, "xsmm": 0.3, "latency": 0.6},
    # Data wrangler: column scans, sorts, streaming.
    "data":         {"cachebwl3": 1.8, "cachebwl2": 1.5, "qs": 1.4,
                     "triad": 1.5, "sleep": 0.7, "xsmm": 0.5},
    # Inference: GEMM-heavy, modest data movement.
    "inference":    {"xsmm": 4.0, "cachebwl3": 1.2, "triad": 1.1,
                     "sleep": 0.5, "intipc": 0.7, "qs": 0.6, "latency": 0.8},
    # Graph analytics: pointer chasing, sparse access, sorting.
    "graph":        {"latency": 3.0, "qs": 1.5, "cachebwl3": 1.2,
                     "sleep": 0.6, "xsmm": 0.5, "intipc": 0.8},
    # Idle/observer: mostly waiting, occasional light work.
    "idle":         {"sleep": 3.0, "cachebwl2": 0.8, "intipc": 0.6,
                     "qs": 0.5, "xsmm": 0.2, "triad": 0.5, "latency": 0.5},
}

# Default role assignment for 8 threads.
DEFAULT_ROLES_8 = [
    "orchestrator",
    "coder",
    "coder",
    "rag",
    "tool",
    "data",
    "inference",
    "graph",
]


def normalise(weights):
    s = sum(weights.values())
    return {k: v / s for k, v in weights.items()}


def role_mix(role, rnd_reps_max):
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


def draw_reps(bench_name, role, rnd_reps_max, rng):
    """Draw a reps multiplier in [1, RND_REPS_MAX] biased per benchmark.

    Agentic workloads have very heterogeneous burst lengths:
      - sleep: mostly short waits, occasional long blocking calls.
      - xsmm:  a few short bursts, occasional long inference batches.
      - intipc/qs/cachebw/triad/latency: medium, somewhat dispersed.
    """
    # Per-bench shape parameters (alpha, beta) for a Beta distribution
    # over [0,1] which we then map to [1, rnd_reps_max].
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

    u = rng.betavariate(a, b)             # u in (0,1)
    k = 1 + int(round(u * (rnd_reps_max - 1)))
    return max(1, min(rnd_reps_max, k))


def gen_thread_schedule(role, rounds, rnd_reps_max, rng):
    mix = role_mix(role, rnd_reps_max)
    schedule = []
    for _ in range(rounds):
        bench = draw_bench(mix, rng)
        k = draw_reps(bench, role, rnd_reps_max, rng)
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
    p.add_argument("--rnd-reps", type=int, default=30,
                   help="Upper bound for the per-round reps multiplier; "
                        "MUST match RND_REPS in AgenticCPUBench.cpp (default: 30).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility (default: 42).")
    p.add_argument("--roles", nargs="*", default=None,
                   help="Optional explicit list of roles, one per thread. "
                        f"Available: {sorted(ROLE_WEIGHTS.keys())}. "
                        "If omitted, a sensible default for 8 threads is used "
                        "and recycled for other thread counts.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-thread mix summary.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.threads <= 0 or args.rounds <= 0 or args.rnd_reps <= 0:
        print("error: --threads, --rounds and --rnd-reps must be positive.",
              file=sys.stderr)
        sys.exit(1)

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
        # Default mapping for 8 threads; for other counts cycle / truncate.
        if args.threads == 8:
            roles = list(DEFAULT_ROLES_8)
        else:
            roles = [DEFAULT_ROLES_8[i % len(DEFAULT_ROLES_8)]
                     for i in range(args.threads)]

    rng = random.Random(args.seed)

    if not args.quiet:
        print(f"Generating {args.threads} schedule(s), {args.rounds} rounds each,"
              f" RND_REPS={args.rnd_reps}, seed={args.seed}")
        print(f"Output files: {args.prefix}_tid0..{args.threads - 1}.sched")
        print()

    out_files = []
    for tid in range(args.threads):
        # Each thread gets its own RNG stream derived from the master seed
        # so individual thread schedules are reproducible independently.
        thread_rng = random.Random(rng.random())
        sched = gen_thread_schedule(roles[tid], args.rounds,
                                    args.rnd_reps, thread_rng)
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
