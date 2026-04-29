#!/usr/bin/env python3
# Copyright (c) 2026, Alexander Heinecke
# analyze_csv.py — Analyse AgenticCPUBench CSV outputs
#
# Usage:
#   python3 analyze_csv.py [<csv_dir_or_files...>]
#
# Examples:
#   python3 analyze_csv.py .              # all *.csv in cwd
#   python3 analyze_csv.py 0.csv 1.csv    # specific files

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse AgenticCPUBench per-thread CSV results."
    )
    p.add_argument(
        "inputs", nargs="*", default=["."],
        help="CSV files or directories containing *.csv files (default: cwd)."
    )
    return p.parse_args()


def collect_csv_files(inputs):
    files = []
    for path in inputs:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "*.csv"))))
        elif os.path.isfile(path):
            files.append(path)
        else:
            print(f"warning: skipping '{path}'", file=sys.stderr)
    return files


def read_samples(csv_files):
    samples = []
    for path in csv_files:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    samples.append({
                        "thread_id":  int(row["thread_id"]),
                        "bench_name": row["bench_name"],
                        "elapsed_s":  float(row["elapsed_s"]),
                        "reps":       int(row["reps"]),
                    })
                except (KeyError, ValueError) as e:
                    print(f"warning: skipping row in {path}: {e}",
                          file=sys.stderr)
    return samples


def analyse(samples):
    total = len(samples)
    if total == 0:
        print("No samples found.")
        return

    threads = sorted({s["thread_id"] for s in samples})

    # Group elapsed times by benchmark name. Each round draws a random
    # rep multiplier k ∈ [1, RND_REPS] (recorded in the `reps` column
    # together with REPS_MULT_*), so the raw elapsed_s values are not
    # directly comparable across invocations of the same benchmark.
    # Normalise to per-rep runtime: norm = elapsed_s / reps.
    by_bench = defaultdict(list)        # raw elapsed_s
    by_bench_norm = defaultdict(list)   # elapsed_s / reps
    for s in samples:
        by_bench[s["bench_name"]].append(s["elapsed_s"])
        if s["reps"] > 0:
            by_bench_norm[s["bench_name"]].append(s["elapsed_s"] / s["reps"])

    bench_names = sorted(by_bench.keys())

    # ── 1. Call fraction per benchmark (across all threads) ──
    print("=" * 72)
    print("  Call Fraction per Benchmark (across all threads)")
    print("=" * 72)
    print(f"  Total samples: {total}   Threads: {len(threads)}")
    print()
    print(f"  {'Benchmark':<14s} {'Count':>7s} {'Fraction':>10s}")
    print(f"  {'-'*14} {'-'*7} {'-'*10}")
    for name in bench_names:
        count = len(by_bench[name])
        frac = count / total
        print(f"  {name:<14s} {count:>7d} {frac:>10.2%}")
    print()

    # ── 2. Expected (mean) runtime per benchmark (per-rep, normalised by `reps`) ──
    print("=" * 72)
    print("  Expected Runtime per Benchmark (normalised per rep)")
    print("=" * 72)
    print("  Each round draws a random k ∈ [1, RND_REPS]; values below are")
    print("  elapsed_s / reps so invocations are directly comparable.")
    print()
    print(f"  {'Benchmark':<14s} {'Count':>7s} {'Mean (s)':>12s} "
          f"{'Min (s)':>12s} {'Max (s)':>12s}")
    print(f"  {'-'*14} {'-'*7} {'-'*12} {'-'*12} {'-'*12}")
    for name in bench_names:
        vals = by_bench_norm[name]
        n = len(vals)
        if n == 0:
            continue
        mean = sum(vals) / n
        print(f"  {name:<14s} {n:>7d} {mean:>12.9f} "
              f"{min(vals):>12.9f} {max(vals):>12.9f}")
    print()

    # ── 3. Runtime variance & standard deviation (per-rep, normalised) ──
    print("=" * 72)
    print("  Runtime Variance per Benchmark (normalised per rep)")
    print("=" * 72)
    print()
    print(f"  {'Benchmark':<14s} {'Count':>7s} {'Mean (s)':>12s} "
          f"{'Var (s²)':>14s} {'StdDev (s)':>14s} {'CoV':>8s}")
    print(f"  {'-'*14} {'-'*7} {'-'*12} {'-'*14} {'-'*14} {'-'*8}")
    for name in bench_names:
        vals = by_bench_norm[name]
        n = len(vals)
        if n == 0:
            continue
        mean = sum(vals) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        else:
            var = 0.0
        std = math.sqrt(var)
        cov = (std / mean * 100) if mean > 0 else 0.0
        print(f"  {name:<14s} {n:>7d} {mean:>12.9f} "
              f"{var:>14.6e} {std:>14.9f} {cov:>7.2f}%")
    print()


def main():
    args = parse_args()
    csv_files = collect_csv_files(args.inputs)

    if not csv_files:
        print("error: no CSV files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {len(csv_files)} CSV file(s): {', '.join(csv_files)}\n")
    samples = read_samples(csv_files)
    analyse(samples)


if __name__ == "__main__":
    main()
