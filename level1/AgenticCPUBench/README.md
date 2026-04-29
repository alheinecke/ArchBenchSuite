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
| `qs` | In-place quicksort (median-of-three pivot, insertion sort ≤16) on int64\_t. Each rep restores the shuffled reference before sorting. | 16 MB | branch-heavy, cache-sensitive | Melements/s |
| `intipc` | Integer IPC stress: 8 independent 64-bit add accumulation chains over a data\[8\]\[4096\] array, inner loop fully unrolled in inline asm. No SIMD. | 1 MB | integer-only | GOPS |
| `latency` | HPCC RandomAccess (GUPS): `table[ran & mask] ^= ran` with LFSR PRNG. Random accesses expose memory latency beyond L3. | 64 MB | random access | GUPS |
| `sleep` | Idle placeholder: each rep calls `usleep(150000)` (0.15 s). Models idle gaps between agent actions. | — | — | — |

### Workloads Approximated by Each Micro-Benchmark

The suite is intentionally small but each kernel is chosen to be a proxy for a
broad class of real-world CPU workloads. The table below maps each micro to
representative workload patterns it stresses similarly.

| Micro | Models what? |
|-------|--------------|
| `sleep` | I/O waits, REST/RPC API calls, external tool calls (MCP, shell sub-processes), LLM round-trips, blocking disk/network reads, lock/condvar waits, polling loops |
| `cachebwl2` / `cachebwl3` | Small-to-medium working-set data processing: array/vector transforms, in-memory table scans, JSON/CSV parsing of cached buffers, columnar projections, tokenisation, log filtering, image-tile/audio-frame processing, hot-loop interpreter dispatch |
| `intipc` | Integer-IPC heavy code: compilers and linkers, JIT/AOT codegen, regex engines, hashing/CRC/checksums, protobuf/flatbuffer (de)serialisation, bytecode interpreters, lexers/parsers, bit-twiddling crypto primitives |
| `qs` | Branch-heavy, cache-sensitive workloads with some memory-latency component: DBMS query execution (sort/merge/hash-join/index probes), OS and filesystem operations (path resolution, b-tree traversal), search and ranking, garbage collectors, route lookups, scheduler decisions |
| `triad` | Large streaming I/O from a single core: file copy / `memcpy`, network buffer drains, `tar`/compression front-ends, large RPC payload (de)serialisation, in-memory shuffle/repartition, video/raw-frame streaming, snapshot/checkpoint writeback |
| `xsmm` | High-IPC SIMD compute (~3.5 IPC class): dense linear algebra, ML inference/training kernels (GEMM/conv via `im2col`), DSP and FFT inner loops, scientific simulations (stencils, BLAS3), embedding/dot-product scoring, batched cosine similarity |
| `latency` (GUPS) | Pointer-chasing and irregular-access workloads: graph processing (BFS/PageRank/connected-components), sparse linear algebra (SpMV), recommendation/embedding-table lookups, KV-store probes, in-memory analytics on large hash tables, symbol-table / scope lookups in interpreters |

### Default Parameters

All working-set sizes are compiled-in constants:

| Benchmark | Working Set Size |
|-----------|-----------------|
| `triad` | 128 MB |
| `cachebwl2` | 512 KB |
| `cachebwl3` | 10 MB |
| `xsmm` | M=64, N=24, K=64, BR=16, F32, alpha=1, beta=0 |
| `qs` | 16 MB |
| `intipc` | 1 MB (num\_iter = 1024 outer iterations) |
| `latency` | 64 MB (largest power-of-two ≤ 64 MiB, num\_updates = 4×n) |
| `sleep` | 0.15 seconds per rep |

### Repetition Multipliers

Each benchmark has a compiled-in repetition multiplier (`REPS_MULT_*`) that
determines how many kernel iterations make up a single round, so that **all
benchmarks run for roughly the same wall-clock time per round**. This ensures
that in `all` mode (random scheduling) each benchmark slot occupies a
comparable time slice regardless of the vastly different per-kernel costs.

The current values were calibrated on a single core of an Intel Core Ultra 7
258V (Lunar Lake, AVX2, WSL) targeting ~0.15–0.25 s per round (measured with
`RND_REPS` temporarily set to 1 so each round runs exactly `REPS_MULT_*`
kernel iterations):

| Benchmark | `REPS_MULT` | ~Time / round (1 thread, 10 rounds) |
|-----------|-------------|-------------------------------------|
| `triad` | 16 | 1.71 s |
| `cachebwl2` | 37 500 | 1.82 s |
| `cachebwl3` | 800 | 2.00 s |
| `xsmm` | 5 000 | 1.79 s |
| `qs` | 1 | 1.61 s |
| `intipc` | 37 | 1.61 s |
| `latency` | 1 | 2.43 s |
| `sleep` | 1 | 1.50 s |

To retune for a different platform, temporarily set `RND_REPS = 1` in
`AgenticCPUBench.cpp`, run each benchmark individually with 10 rounds, then
adjust the multipliers until all benchmarks converge to the same total time.
Restore `RND_REPS = 30` afterwards.

### Per-Round Random Reps Multiplier (`RND_REPS`)

In addition to the fixed per-benchmark `REPS_MULT_*` values, each round draws
an independent random integer `k ∈ [1, RND_REPS]` (compile-time constant,
currently `RND_REPS = 30`) and runs the selected benchmark with `k` passed
as the rep count (which the benchmark internally multiplies by `REPS_MULT_*`).
This adds temporal variability to the workload, better modelling the
unpredictable burst lengths of agent actions.

The drawn `k` value for every round is written to the schedule file by
`--dump-schedule` and restored by `--replay-schedule`, so reruns are
bit-for-bit reproducible.

## Usage

```
./AgenticCPUBench_<ISA>.exe <benchmark> <rounds> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `benchmark` | Which benchmark(s) to enable. One of: `triad`, `cachebwl2`, `cachebwl3`, `xsmm`, `qs`, `intipc`, `latency`, `sleep`, or **`all`** (enables all eight). |
| `rounds` | Number of outer rounds. Each round, **every OpenMP thread independently picks one benchmark at random** from the enabled set and runs it with a randomised reps multiplier `k ∈ [1, RND_REPS]`. |

### Options

| Option | Description |
|--------|-------------|
| `--dump-schedule <prefix>` | Write each thread's random schedule to `<prefix>_tid<N>.sched`. Each line contains two integers: the benchmark index and the per-round random reps multiplier drawn from `[1, RND_REPS]`. |
| `--replay-schedule <prefix>` | Read schedules from `<prefix>_tid<N>.sched` and replay them exactly (both benchmark choice and reps multiplier), instead of drawing random values. |

### Environment Variables

| Variable | Effect |
|----------|--------|
| `OMP_NUM_THREADS` | Number of parallel threads (one benchmark instance per thread). |
| `LD_LIBRARY_PATH` | Must include `./libxsmm/lib` (or wherever `libxsmm.so` is installed) at runtime. |

### Examples

```bash
# Single benchmark, 1 thread, 10 rounds
LD_LIBRARY_PATH=./libxsmm/lib OMP_NUM_THREADS=1 ./AgenticCPUBench_avx2.exe triad 10

# All benchmarks, 8 threads, 100 rounds (randomised per-thread)
LD_LIBRARY_PATH=./libxsmm/lib OMP_NUM_THREADS=8 ./AgenticCPUBench_avx2.exe all 100

# AVX-512 build, 4 threads, 50 rounds
LD_LIBRARY_PATH=./libxsmm/lib OMP_NUM_THREADS=4 ./AgenticCPUBench_avx512.exe all 50

# Dump the random schedule for later replay
LD_LIBRARY_PATH=./libxsmm/lib OMP_NUM_THREADS=4 ./AgenticCPUBench_avx2.exe all 100 --dump-schedule ./my_run

# Replay a previously saved schedule (exact same benchmark order per thread)
LD_LIBRARY_PATH=./libxsmm/lib OMP_NUM_THREADS=4 ./AgenticCPUBench_avx2.exe all 100 --replay-schedule ./my_run
```

### Generating an Agentic Workload Schedule

The helper script `gen_agentic_schedule.py` synthesises per-thread schedule
files that approximate an agentic-AI worker mix on multiple cores. Each
thread is assigned a *role* (orchestrator, coder, rag, tool, data,
inference, graph, idle) which biases its benchmark mix and per-round
reps multiplier (`k ∈ [1, RND_REPS]`) toward the workloads listed in
"Workloads Approximated by Each Micro-Benchmark":

- `sleep` bursts are heavy-tailed (lots of short waits, occasional long
  blocking I/O / LLM round-trips).
- `xsmm` reps spike on `inference`-role threads (long GEMM batches).
- `intipc` and `qs` dominate `coder` threads (compilation, search, sort).
- `latency` is amplified for `rag` and `graph` roles (embedding / KV /
  graph traversal).

```bash
# Generate 8-thread schedules, 200 rounds each, for the default agentic mix
python3 gen_agentic_schedule.py --threads 8 --rounds 200 --prefix agentic

# Replay them
LD_LIBRARY_PATH=./libxsmm/lib OMP_NUM_THREADS=8 \
  ./AgenticCPUBench_avx2.exe all 200 --replay-schedule agentic

# Customise: pin specific roles per thread
python3 gen_agentic_schedule.py --threads 8 --rounds 500 --prefix infserve \
    --roles inference inference inference inference \
            rag        rag        tool       orchestrator
```

The script honours `RND_REPS` (default 30); pass `--rnd-reps N` if you
have changed the compile-time constant in `AgenticCPUBench.cpp`.

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

Example output (8 threads, 15 rounds each, `all` mode on Intel Core Ultra 7 258V):

```
========================================================================
  Call Fraction per Benchmark (across all threads)
========================================================================
  Total samples: 120   Threads: 8

  Benchmark        Count   Fraction
  -------------- ------- ----------
  cachebwl2           13     10.83%
  cachebwl3           12     10.00%
  intipc              13     10.83%
  latency             19     15.83%
  qs                  12     10.00%
  sleep               20     16.67%
  triad               14     11.67%
  xsmm                17     14.17%

========================================================================
  Expected Runtime per Benchmark (normalised per rep)
========================================================================
  Each round draws a random k ∈ [1, RND_REPS]; values below are
  elapsed_s / reps so invocations are directly comparable.

  Benchmark        Count     Mean (s)      Min (s)      Max (s)
  -------------- ------- ------------ ------------ ------------
  cachebwl2           13  0.000009973  0.000007199  0.000011839
  cachebwl3           12  0.000608011  0.000295176  0.000791938
  intipc              13  0.007185046  0.004838574  0.009250811
  latency             19  0.444615762  0.240411833  0.609337000
  qs                  12  0.284511370  0.206190500  0.369728444
  sleep               20  0.150335620  0.150125889  0.152118412
  triad               14  0.020667255  0.015667931  0.025096090
  xsmm                17  0.000066878  0.000036811  0.000086406

========================================================================
  Runtime Variance per Benchmark (normalised per rep)
========================================================================

  Benchmark        Count     Mean (s)       Var (s²)     StdDev (s)      CoV
  -------------- ------- ------------ -------------- -------------- --------
  cachebwl2           13  0.000009973   2.005677e-12    0.000001416   14.20%
  cachebwl3           12  0.000608011   2.937051e-08    0.000171378   28.19%
  intipc              13  0.007185046   1.693730e-06    0.001301434   18.11%
  latency             19  0.444615762   7.896557e-03    0.088862576   19.99%
  qs                  12  0.284511370   2.761424e-03    0.052549248   18.47%
  sleep               20  0.150335620   2.060314e-07    0.000453907    0.30%
  triad               14  0.020667255   9.927121e-06    0.003150733   15.25%
  xsmm                17  0.000066878   2.178559e-10    0.000014760   22.07%

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
