# AgenticCPUBench — Randomised Mixed Micro-Benchmark Suite

## Motivation: Why a CPU Benchmark for the Agentic AI Era?

The rise of agentic AI systems is fundamentally changing how CPUs are utilised in
data-centre and client environments. Unlike traditional batch inference, where a
GPU saturates a single workload at a time, agentic AI orchestrates *many
heterogeneous actions concurrently on the CPU*.

Modern AI agent frameworks—such as Microsoft's Magentic-One (AutoGen), Anthropic's
tool-using Claude agents, LangChain/LangGraph pipelines, and numerous
coding-assistant agents (GitHub Copilot, Cursor, etc.)—follow a common execution
pattern:

1. **An orchestrator** decomposes a high-level goal into sub-tasks, assigns them
   to specialised worker agents, and tracks progress through outer and inner
   control loops.
2. **Worker agents** execute diverse CPU-side actions: running shell commands and
   compilers, parsing files and build outputs, performing text search and
   retrieval (grep, semantic search, embeddings), sorting and ranking candidate
   results, executing Python/Node scripts, and managing data structures.
3. **Multiple agents run in parallel** on separate cores, each performing a
   different kind of work at any given moment. One core may be running a compiler
   (integer IPC–heavy), another may be searching a large text corpus (memory
   bandwidth–bound), a third may be sorting candidate results (branch-heavy,
   cache-sensitive), and a fourth may be idle between LLM round-trips.

This workload has several distinctive properties that existing benchmarks fail to
capture:

- **Heterogeneity across cores.** At any instant, different cores execute
  fundamentally different micro-kernels (FP STREAM-like sweeps, integer
  accumulation, random-access latency probes, branch-heavy sorts, GEMM
  compute). Traditional benchmarks run the *same* workload on every core.
- **Temporal randomness.** Agent steps arrive in unpredictable order because they
  depend on LLM reasoning output. There is no static schedule; each core's next
  task is essentially a random draw from the set of possible operations.
- **Mixed arithmetic intensity.** Agent actions span the full roofline—from pure
  memory-bandwidth operations (file I/O, text scanning) through moderate
  intensity (data transformations) to compute-dense kernels (embedding
  generation, local model inference via GEMM).
- **Idle gaps.** Agents frequently wait for external I/O (LLM API responses,
  network fetches, disk reads). Cores transition between full load and idle,
  stressing the CPU's power management, frequency scaling, and cache warm-up
  behaviour.

**AgenticCPUBench** models this workload pattern directly: each OpenMP thread
independently draws a random micro-benchmark from the enabled set on every round
and executes it. This creates exactly the kind of heterogeneous, temporally random,
mixed-intensity load that agentic AI systems impose on CPUs. It is therefore a
useful first-level proxy benchmark for evaluating CPU suitability for agentic AI
server and client workloads.

## Benchmarks

| Name | Description | Working Set | Arithmetic Intensity | Key Metric |
|------|-------------|-------------|---------------------|------------|
| `triad` | STREAM Triad: `A[i] = B[i] + s·C[i]` with platform-specific SIMD intrinsics (AVX-512, AVX2, SSE, NEON). Streaming stores (`_mm*_stream_pd`). | 128 MB | ~0.083 FLOP/byte | bandwidth (GB/s) |
| `cachebwl2` | Sequential SIMD read sweep over a small array (targets L2 cache). | 512 KB | 0 (pure read) | bandwidth (GB/s) |
| `cachebwl3` | Sequential SIMD read sweep over a medium array (targets L3 cache). | 10 MB | 0 (pure read) | bandwidth (GB/s) |
| `xsmm` | F32 strided batch-reduce GEMM via libxsmm JIT. M=64, N=24, K=64, BR=16. | ~3 MB | high | GFLOPS |
| `qs` | In-place quicksort (median-of-three pivot, insertion sort ≤16) on int64\_t. Each rep restores the shuffled reference before sorting. | 64 MB | branch-heavy, cache-sensitive | Melements/s |
| `intipc` | Integer IPC stress: 8 independent 64-bit add accumulation chains over a data\[8\]\[4096\] array, inner loop fully unrolled in inline asm. No SIMD. | 1 MB | integer-only | GOPS |
| `latency` | HPCC RandomAccess (GUPS): `table[ran & mask] ^= ran` with LFSR PRNG. Random accesses expose memory latency beyond L3. | 64 MB | random access | GUPS |
| `sleep` | Idle placeholder: each rep calls `usleep(500000)` (0.5 s). Models idle gaps between agent actions. | — | — | — |

### Default Parameters

All working-set sizes are compiled-in constants:

| Benchmark | Working Set Size |
|-----------|-----------------|
| `triad` | 128 MB |
| `cachebwl2` | 512 KB |
| `cachebwl3` | 10 MB |
| `xsmm` | M=64, N=24, K=64, BR=16, F32, alpha=1, beta=0 |
| `qs` | 64 MB |
| `intipc` | 1 MB (num\_iter = 1024 outer iterations) |
| `latency` | 64 MB (largest power-of-two ≤ 64 MiB, num\_updates = 4×n) |
| `sleep` | 0.5 seconds per rep |

### Repetition Multipliers

Each benchmark has a compiled-in repetition multiplier (`REPS_MULT_*`) that
scales the user-supplied repetition count so that **all benchmarks run for
roughly the same wall-clock time per round** when invoked with `reps=1`.
This ensures that in `all` mode (random scheduling) each benchmark slot
occupies a comparable time slice regardless of the vastly different per-kernel
costs.

The current values were calibrated on a single core of an Intel Core Ultra 7
258V (Lunar Lake, AVX2, WSL) targeting ~0.6 s per round:

| Benchmark | `REPS_MULT` | ~Time / round (1 thread, 10 rounds) |
|-----------|-------------|-------------------------------------|
| `triad` | 64 | 6.10 s |
| `cachebwl2` | 150 000 | 6.34 s |
| `cachebwl3` | 3 000 | 6.62 s |
| `xsmm` | 20 000 | 6.12 s |
| `qs` | 1 | 6.58 s |
| `intipc` | 150 | 5.52 s |
| `latency` | 3 | 6.69 s |
| `sleep` | 1 | 5.00 s |

To retune for a different platform, run each benchmark individually with
`reps=1` and 10 rounds, then adjust the multipliers in `AgenticCPUBench.cpp`
until all benchmarks converge to the same total time.

## Usage

```
./AgenticCPUBench_<ISA>.exe <benchmark> <repetitions> <rounds> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `benchmark` | Which benchmark(s) to enable. One of: `triad`, `cachebwl2`, `cachebwl3`, `xsmm`, `qs`, `intipc`, `latency`, `sleep`, or **`all`** (enables all eight). |
| `repetitions` | Number of timed kernel repetitions per benchmark invocation within a single round. |
| `rounds` | Number of outer rounds. Each round, **every OpenMP thread independently picks one benchmark at random** from the enabled set and runs it for the given number of repetitions. |

### Options

| Option | Description |
|--------|-------------|
| `--dump-schedule <prefix>` | Write each thread's random schedule to `<prefix>_tid<N>.sched` (one benchmark index per line). |
| `--replay-schedule <prefix>` | Read schedules from `<prefix>_tid<N>.sched` and replay them exactly, instead of drawing random choices. |

### Environment Variables

| Variable | Effect |
|----------|--------|
| `OMP_NUM_THREADS` | Number of parallel threads (one benchmark instance per thread). |
| `LD_LIBRARY_PATH` | Must include `./libxsmm/lib` (or wherever `libxsmm.so` is installed) at runtime. |

### Examples

```bash
# Single benchmark, 1 thread, 10 rounds
LD_LIBRARY_PATH=./libxsmm/lib OMP_NUM_THREADS=1 ./AgenticCPUBench_avx2.exe triad 1 10

# All benchmarks, 8 threads, 100 rounds (randomised per-thread)
LD_LIBRARY_PATH=./libxsmm/lib OMP_NUM_THREADS=8 ./AgenticCPUBench_avx2.exe all 1 100

# AVX-512 build, 4 threads
LD_LIBRARY_PATH=./libxsmm/lib OMP_NUM_THREADS=4 ./AgenticCPUBench_avx512.exe all 2 50

# Dump the random schedule for later replay
LD_LIBRARY_PATH=./libxsmm/lib OMP_NUM_THREADS=4 ./AgenticCPUBench_avx2.exe all 1 100 --dump-schedule ./my_run

# Replay a previously saved schedule (exact same benchmark order per thread)
LD_LIBRARY_PATH=./libxsmm/lib OMP_NUM_THREADS=4 ./AgenticCPUBench_avx2.exe all 1 100 --replay-schedule ./my_run
```

## Output

### CSV Files

Each run produces a CSV file (`<tid>.csv`) per thread containing a merged,
monotonic-time-sorted timeline of all benchmark samples:

```
host,thread_id,bench_name,reps,monotonic_time_s,elapsed_s,<metrics...>
```

| Column | Description |
|--------|-------------|
| `host` | Hostname of the machine. |
| `thread_id` | OpenMP thread number. |
| `bench_name` | Name of the benchmark (e.g. `triad`, `xsmm`). |
| `reps` | Number of kernel repetitions in this invocation. |
| `monotonic_time_s` | `CLOCK_MONOTONIC` timestamp at measurement start (seconds). Shared across all threads, so samples from different CSVs are directly comparable. |
| `elapsed_s` | Wall-clock duration of the benchmark invocation (seconds). |

Additional metric columns vary by benchmark (e.g. `bandwidth_GBs`, `GFLOPS`,
`Melements_per_s`, `GUPS`, `GOPS`). Missing metrics for a given benchmark appear
as empty fields.

### Perfetto Trace Visualisation

The included `csv_to_perfetto.py` script converts the per-thread CSV files into
a single Perfetto JSON trace that can be visualised in Chrome.

```bash
# Convert all CSVs in the current directory
python3 csv_to_perfetto.py .

# Convert specific files with a custom output name
python3 csv_to_perfetto.py -o my_trace.json 0.csv 1.csv 2.csv

# Convert all CSVs in another directory
python3 csv_to_perfetto.py /path/to/results/
```

Open the resulting `trace.json` at `chrome://tracing` or
[ui.perfetto.dev](https://ui.perfetto.dev). The trace shows:

- **One row per thread** — each OpenMP thread is a separate timeline row.
- **One event per benchmark invocation** — duration matches `elapsed_s`.
- **Colour-coded benchmarks** — each benchmark type has a distinct colour (e.g. triad = green, latency = red, xsmm = orange).
- **Time-aligned threads** — all threads share the `monotonic_time_s` clock, so events are correctly aligned across rows.
- **Metric tooltips** — hover over any event to see bandwidth, GFLOPS, array size, and other metrics in the detail panel.

### Statistical Analysis

The `analyze_csv.py` script summarises benchmark results across all threads:

```bash
# Analyse all CSVs in the current directory
python3 analyze_csv.py .

# Analyse specific files
python3 analyze_csv.py 0.csv 1.csv 2.csv
```

It prints three tables:

1. **Call Fraction** — how often each benchmark was selected across all threads (verifies uniform random scheduling).
2. **Expected Runtime** — mean, min, and max elapsed time per benchmark.
3. **Runtime Variance** — variance, standard deviation, and coefficient of variation (CoV) per benchmark.

Example output (8 threads, 50 rounds each, `all` mode on Intel Core Ultra 7 258V):

```
========================================================================
  Call Fraction per Benchmark (across all threads)
========================================================================
  Total samples: 400   Threads: 8

  Benchmark        Count   Fraction
  -------------- ------- ----------
  cachebwl2           48     12.00%
  cachebwl3           49     12.25%
  intipc              48     12.00%
  latency             59     14.75%
  qs                  49     12.25%
  sleep               45     11.25%
  triad               53     13.25%
  xsmm                49     12.25%

========================================================================
  Expected Runtime per Benchmark
========================================================================

  Benchmark        Count   Mean (s)    Min (s)    Max (s)
  -------------- ------- ---------- ---------- ----------
  cachebwl2           48   1.235370   1.028665   1.669207
  cachebwl3           49   1.635694   1.117664   2.056558
  intipc              48   0.952151   0.797778   1.188861
  latency             59   1.291113   0.844007   1.557255
  qs                  49   1.040602   0.865494   1.420287
  sleep               45   0.500276   0.500084   0.506103
  triad               53   1.225150   0.905206   1.429427
  xsmm                49   1.254924   1.054650   1.719608

========================================================================
  Runtime Variance per Benchmark
========================================================================

  Benchmark        Count   Mean (s)     Var (s²)   StdDev (s)      CoV
  -------------- ------- ---------- ------------ ------------ --------
  cachebwl2           48   1.235370  0.018524423     0.136104   11.02%
  cachebwl3           49   1.635694  0.040906540     0.202254   12.37%
  intipc              48   0.952151  0.007863225     0.088675    9.31%
  latency             59   1.291113  0.026411390     0.162516   12.59%
  qs                  49   1.040602  0.012370990     0.111225   10.69%
  sleep               45   0.500276  0.000000790     0.000889    0.18%
  triad               53   1.225150  0.014465213     0.120271    9.82%
  xsmm                49   1.254924  0.016929074     0.130112   10.37%
```

## Building

### Prerequisites

- g++ (or clang++) with C++11 and OpenMP support
- libxsmm (automatically cloned and built from source if not present)

### Build Commands

```bash
# Default: build both AVX2 and AVX-512 binaries
make

# Explicitly build all x86 variants
make all_x86

# Build a single ISA variant
make ISA=avx2   AgenticCPUBench_avx2.exe
make ISA=avx512 AgenticCPUBench_avx512.exe
make ISA=aarch64 AgenticCPUBench_aarch64.exe

# Clean build artifacts
make clean

# Clean everything including cloned libxsmm
make distclean
```

The default target (`all`) is equivalent to `all_x86`, which builds both
`AgenticCPUBench_avx2.exe` and `AgenticCPUBench_avx512.exe`.

### ISA-Specific Flags

| ISA | Compiler Flags |
|-----|---------------|
| `avx2` | `-mavx2` |
| `avx512` | `-mavx512f -mavx512cd -mavx512bw -mavx512dq` |
| `aarch64` | `-march=armv8-a` |

## Architecture

```
AgenticCPUBench.cpp       Main: parse args, OpenMP parallel region, random dispatch
microbench.hpp            Abstract base class (MicroBench), PerfSample struct, CSV dump
triad_bench.hpp/.cpp      STREAM Triad with SIMD intrinsics
cachebw_bench.hpp/.cpp    Cache bandwidth with inline asm SIMD loads
xsmm_bench.hpp/.cpp       libxsmm BRGEMM JIT kernel
qs_bench.hpp/.cpp         Quicksort benchmark
intipc_bench.hpp/.cpp     Integer IPC stress (inline asm)
latency_bench.hpp/.cpp    HPCC RandomAccess / GUPS
sleep_bench.hpp           Idle placeholder (header-only)
```

Each benchmark implements the `MicroBench` interface:

```cpp
virtual void setup_benchmark(size_t size_bytes) = 0;
virtual void run_benchmark(int reps) = 0;
virtual void destroy_benchmark() = 0;
```

Setup is called once before the rounds loop. Each `run_benchmark` call records a
`PerfSample` with a monotonic timestamp, elapsed time, and benchmark-specific
metrics. After all rounds complete, `dump_merged_perf_data()` writes the
time-sorted CSV.

## License

BSD 3-Clause. See individual source files for the full copyright notice.
