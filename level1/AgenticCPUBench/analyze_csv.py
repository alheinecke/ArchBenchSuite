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

    # Group elapsed times by benchmark name
    by_bench = defaultdict(list)
    for s in samples:
        by_bench[s["bench_name"]].append(s["elapsed_s"])

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

    # ── 2. Expected (mean) runtime per benchmark ──
    print("=" * 72)
    print("  Expected Runtime per Benchmark")
    print("=" * 72)
    print()
    print(f"  {'Benchmark':<14s} {'Count':>7s} {'Mean (s)':>10s} "
          f"{'Min (s)':>10s} {'Max (s)':>10s}")
    print(f"  {'-'*14} {'-'*7} {'-'*10} {'-'*10} {'-'*10}")
    for name in bench_names:
        vals = by_bench[name]
        n = len(vals)
        mean = sum(vals) / n
        print(f"  {name:<14s} {n:>7d} {mean:>10.6f} "
              f"{min(vals):>10.6f} {max(vals):>10.6f}")
    print()

    # ── 3. Runtime variance & standard deviation ──
    print("=" * 72)
    print("  Runtime Variance per Benchmark")
    print("=" * 72)
    print()
    print(f"  {'Benchmark':<14s} {'Count':>7s} {'Mean (s)':>10s} "
          f"{'Var (s²)':>12s} {'StdDev (s)':>12s} {'CoV':>8s}")
    print(f"  {'-'*14} {'-'*7} {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
    for name in bench_names:
        vals = by_bench[name]
        n = len(vals)
        mean = sum(vals) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        else:
            var = 0.0
        std = math.sqrt(var)
        cov = (std / mean * 100) if mean > 0 else 0.0
        print(f"  {name:<14s} {n:>7d} {mean:>10.6f} "
              f"{var:>12.9f} {std:>12.6f} {cov:>7.2f}%")
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
