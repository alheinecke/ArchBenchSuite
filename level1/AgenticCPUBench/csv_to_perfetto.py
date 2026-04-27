#!/usr/bin/env python3
# Copyright (c) 2026, Alexander Heinecke
# csv_to_perfetto.py — Convert AgenticCPUBench CSV files to Perfetto JSON trace
#
# Usage:
#   python3 csv_to_perfetto.py [--output trace.json] <csv_dir_or_files...>
#
# Examples:
#   python3 csv_to_perfetto.py .                          # all *.csv in cwd
#   python3 csv_to_perfetto.py 0.csv 1.csv 2.csv          # explicit files
#   python3 csv_to_perfetto.py --output my_trace.json .    # custom output name
#
# Open the resulting JSON in Chrome via  chrome://tracing  or  https://ui.perfetto.dev

import argparse
import csv
import glob
import json
import os
import sys


# Perfetto cname palette — one colour per benchmark
_BENCH_COLORS = {
    "triad":     "thread_state_running",        # green
    "cachebwl2": "rail_response",               # blue
    "cachebwl3": "yellow",                      # yellow
    "xsmm":      "thread_state_iowait",         # orange
    "qs":        "generic_work",                 # purple
    "intipc":    "good",                         # olive
    "latency":   "terrible",                     # red
    "sleep":     "thread_state_sleeping",        # grey
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert AgenticCPUBench per-thread CSVs to a Perfetto JSON trace."
    )
    p.add_argument(
        "inputs", nargs="+",
        help="CSV files or directories containing *.csv files."
    )
    p.add_argument(
        "-o", "--output", default="trace.json",
        help="Output JSON filename (default: trace.json)."
    )
    return p.parse_args()


def collect_csv_files(inputs):
    """Resolve directories and globs into a list of CSV file paths."""
    files = []
    for path in inputs:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "*.csv"))))
        elif os.path.isfile(path):
            files.append(path)
        else:
            print(f"warning: skipping '{path}' (not a file or directory)",
                  file=sys.stderr)
    return files


def read_samples(csv_files):
    """Read all CSV files and return a list of sample dicts."""
    samples = []
    for path in csv_files:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    samples.append({
                        "host":       row["host"],
                        "thread_id":  int(row["thread_id"]),
                        "bench_name": row["bench_name"],
                        "reps":       int(row["reps"]),
                        "mono_s":     float(row["monotonic_time_s"]),
                        "elapsed_s":  float(row["elapsed_s"]),
                        "raw":        row,
                    })
                except (KeyError, ValueError) as e:
                    print(f"warning: skipping row in {path}: {e}",
                          file=sys.stderr)
    return samples


def build_trace_events(samples):
    """
    Build Perfetto Trace-Event-Format (JSON) events.

    Each benchmark invocation becomes a 'Complete' (ph='X') event whose
    timestamp (ts) is derived from monotonic_time_s and duration (dur)
    from elapsed_s.  We subtract the global minimum monotonic time so the
    trace starts near zero.

    Durations are clamped per-thread so that no event extends past the
    start of the next event on the same track (avoids Perfetto
    slice_drop_overlapping_complete_event errors).
    """
    if not samples:
        return []

    # Global time origin (seconds) — earliest monotonic timestamp
    t_origin = min(s["mono_s"] for s in samples)

    # Group by thread and sort by monotonic time within each thread
    from collections import defaultdict
    by_tid = defaultdict(list)
    for s in samples:
        by_tid[s["thread_id"]].append(s)
    for tid in by_tid:
        by_tid[tid].sort(key=lambda s: s["mono_s"])

    events = []
    for tid, thread_samples in sorted(by_tid.items()):
        for i, s in enumerate(thread_samples):
            ts_us  = (s["mono_s"] - t_origin) * 1e6   # microseconds
            dur_us = s["elapsed_s"] * 1e6

            # Clamp duration so it does not overlap the next event on this thread
            if i + 1 < len(thread_samples):
                next_ts_us = (thread_samples[i + 1]["mono_s"] - t_origin) * 1e6
                max_dur = next_ts_us - ts_us
                if max_dur > 0 and dur_us > max_dur:
                    dur_us = max_dur

            # Build a compact tooltip with the most interesting metrics
            args = {"reps": s["reps"]}
            for key in ("bandwidth_GBs", "GFLOPS", "elapsed_s",
                         "array_size_MB", "n_elements",
                         "arithmetic_intensity_FLOP_byte"):
                val = s["raw"].get(key, "")
                if val:
                    try:
                        args[key] = float(val)
                    except ValueError:
                        args[key] = val

            event = {
                "name": s["bench_name"],
                "cat":  "benchmark",
                "ph":   "X",           # Complete event
                "ts":   ts_us,
                "dur":  dur_us,
                "pid":  1,             # single process
                "tid":  s["thread_id"] + 1,
                "args": args,
            }

            cname = _BENCH_COLORS.get(s["bench_name"])
            if cname:
                event["cname"] = cname

            events.append(event)

    return events


def build_metadata(samples):
    """Thread-name and process-name metadata events."""
    meta = [
        {"name": "process_name", "ph": "M", "pid": 1, "tid": 0,
         "args": {"name": "AgenticCPUBench"}},
    ]
    tids = sorted({s["thread_id"] for s in samples})
    for tid in tids:
        meta.append({
            "name": "thread_name", "ph": "M", "pid": 1, "tid": tid + 1,
            "args": {"name": f"Thread {tid}"},
        })
    return meta


def main():
    args = parse_args()
    csv_files = collect_csv_files(args.inputs)

    if not csv_files:
        print("error: no CSV files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {len(csv_files)} CSV file(s): {', '.join(csv_files)}")
    samples = read_samples(csv_files)
    print(f"  {len(samples)} sample(s) across "
          f"{len({s['thread_id'] for s in samples})} thread(s)")

    events = build_trace_events(samples)
    metadata = build_metadata(samples)

    trace = {"traceEvents": metadata + events}

    with open(args.output, "w") as f:
        json.dump(trace, f)

    print(f"Wrote {args.output} ({len(events)} events)")
    print(f"Open in Chrome:  chrome://tracing  or  https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
