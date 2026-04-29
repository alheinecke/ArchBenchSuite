/******************************************************************************
** Copyright (c) 2026, Alexander Heinecke                                    **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/

/* AgenticCPUBench: combines memory-bandwidth and FP-compute in a single kernel
** to explore the roofline model at various arithmetic intensities.
**
** Benchmarks are selected via the MicroBench interface; currently runs:
**
**   TriadBench:  A[i] = B[i] + scalar * C[i]  –  fixed 512 MB
**     - 2 loads + 1 store  => 3 * sizeof(double) bytes per element
**     - 2 FP ops per element
**     => arithmetic intensity ~ 2 / (3*8) = 0.083 FLOP/byte
**
**   CacheBwBench (two working-set variants):
**     cachebwl2  –  512 KB  (targets L2 cache)
**     cachebwl3  –  10 MB   (targets L3 cache)
**     - 1 load per element => sizeof(double) bytes per element
**     - 0 FP ops  => arithmetic intensity = 0 FLOP/byte (pure bandwidth)
**
**   XsmmBench: F32 strided BRGEMM via libxsmm JIT (M=64, N=24, K=64, BR=16)
**     - alpha=1, beta=0, LIBXSMM_GEMM_BATCH_REDUCE_STRIDE
**     - FLOPS per call: 2 * 64 * 24 * 64 * 16 = 3,145,728
**
**   QsBench: in-place quicksort on an array of int64_t values  –  fixed 16 MB
**     - median-of-three pivot, insertion sort below 16 elements
**     - each rep restores the pre-shuffled reference before timing the sort
**     - reported metric: throughput in million elements per second (Melements/s)
**
**   IntIpcBench: integer IPC stress via scalar 64-bit add accumulation  –  1 MB scale
**     - data[M=8][N=4096] int64_t, accum[8]; inner j-loop fully unrolled
**     - num_iter = (size_bytes / MiB) * 1024 outer iterations
**     - x86_64: addq (mem), reg  (8 independent add-chains)
**     - AArch64: ldr + add scalar integer (8 independent add-chains)
**     - no SSE/AVX/NEON/SVE instructions
**     - reported metric: GOPS (10^9 integer adds per second)
**
**   LatencyBench: idle memory-latency benchmark (HPCC RandomAccess / GUPS)  –  256 MB
**     - table of uint64_t, size = largest power-of-two <= 256 MiB
**     - HPCC LFSR PRNG: ran = (ran<<1) ^ (msb_set ? POLY : 0), POLY=7
**     - kernel: table[ran & mask] ^= ran,  num_updates = 4 * n
**     - random accesses cause L3/DRAM cache misses -> exposes memory latency
**     - reported metric: GUPS (10^9 updates per second)
**
**   SleepBench: idle pseudo-benchmark
**     - each repetition calls sleep(1) for exactly 1 second
**     - useful as an idle slot between active benchmarks in random rounds
**
** Usage: AgenticCPUBench <benchmark> <rounds>
**   benchmark: triad | cachebwl2 | cachebwl3 | xsmm | qs | intipc | latency | sleep | all
**   rounds: number of outer rounds; each round runs one randomly chosen
**           benchmark with a randomly drawn reps multiplier in [1, RND_REPS]
*/

#include "triad_bench.hpp"
#include "cachebw_bench.hpp"
#include "xsmm_bench.hpp"
#include "qs_bench.hpp"
#include "intipc_bench.hpp"
#include "latency_bench.hpp"
#include "sleep_bench.hpp"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

/* Fixed working-set sizes (in bytes) for each benchmark */
static constexpr size_t SIZE_TRIAD      = 128ULL * 1024 * 1024; /* 128 MB */
static constexpr size_t SIZE_CACHEBWL2  = 512ULL * 1024;        /* 512 KB */
static constexpr size_t SIZE_CACHEBWL3  = 10ULL * 1024 * 1024;  /*  10 MB */
static constexpr size_t SIZE_QS         = 16ULL * 1024 * 1024;  /*  16 MB */
static constexpr size_t SIZE_INTIPC     = 1ULL * 1024 * 1024;   /*   1 MB */
static constexpr size_t SIZE_LATENCY    = 64ULL * 1024 * 1024;  /*  64 MB */

/* Fixed multipliers applied to the user-supplied repetition count */
/* They have fudged-factor on a single core of Intel(R) Core(TM) Ultra 7 258V in WSL */
static constexpr int REPS_MULT_TRIAD     = 16;
static constexpr int REPS_MULT_CACHEBWL2 = 37500;
static constexpr int REPS_MULT_CACHEBWL3 = 800;
static constexpr int REPS_MULT_QS        = 1;
static constexpr int REPS_MULT_INTIPC    = 37;
static constexpr int REPS_MULT_LATENCY   = 1;
static constexpr int REPS_MULT_XSMM      = 5000;
static constexpr int REPS_MULT_SLEEP     = 1;

/* Upper bound (inclusive) for the per-round random reps multiplier drawn
 * uniformly from [1, RND_REPS]. */
static constexpr int RND_REPS = 30;

/* Pair of display name + benchmark instance */
struct BenchEntry {
    std::string   name;
    MicroBench*   bench;
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0]
                  << " <benchmark> <rounds> [options]" << std::endl;
        std::cout << "  benchmark: triad | cachebwl2 | cachebwl3 | xsmm | qs | intipc | latency | sleep | all" << std::endl;
        std::cout << "  options:" << std::endl;
        std::cout << "    --dump-schedule <prefix>    write per-thread schedule to <prefix>_tid<N>.sched" << std::endl;
        std::cout << "    --replay-schedule <prefix>  replay schedule from <prefix>_tid<N>.sched" << std::endl;
        return -1;
    }

    const char* bench_name = argv[1];
    const int   rounds     = atoi(argv[2]);

    /* Parse optional flags */
    std::string dump_prefix;
    std::string replay_prefix;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--dump-schedule") == 0 && i + 1 < argc) {
            dump_prefix = argv[++i];
        } else if (strcmp(argv[i], "--replay-schedule") == 0 && i + 1 < argc) {
            replay_prefix = argv[++i];
        } else {
            std::cerr << "Error: unknown option '" << argv[i] << "'" << std::endl;
            return -1;
        }
    }

    const bool run_triad     = (strcmp(bench_name, "triad")     == 0 ||
                                 strcmp(bench_name, "all")       == 0);
    const bool run_cachebwl2 = (strcmp(bench_name, "cachebwl2") == 0 ||
                                 strcmp(bench_name, "all")       == 0);
    const bool run_cachebwl3 = (strcmp(bench_name, "cachebwl3") == 0 ||
                                 strcmp(bench_name, "all")       == 0);
    const bool run_xsmm      = (strcmp(bench_name, "xsmm")      == 0 ||
                                 strcmp(bench_name, "all")       == 0);
    const bool run_qs        = (strcmp(bench_name, "qs")         == 0 ||
                                 strcmp(bench_name, "all")       == 0);
    const bool run_intipc    = (strcmp(bench_name, "intipc")     == 0 ||
                                 strcmp(bench_name, "all")       == 0);
    const bool run_latency   = (strcmp(bench_name, "latency")    == 0 ||
                                 strcmp(bench_name, "all")       == 0);
    const bool run_sleep     = (strcmp(bench_name, "sleep")      == 0 ||
                                 strcmp(bench_name, "all")       == 0);

    if (!run_triad && !run_cachebwl2 && !run_cachebwl3 &&
        !run_xsmm  && !run_qs        && !run_intipc     && !run_latency && !run_sleep) {
        std::cerr << "Error: unknown benchmark '" << bench_name
                  << "'.\nChoose: triad | cachebwl2 | cachebwl3 | xsmm | qs | intipc | latency | sleep | all"
                  << std::endl;
        return -1;
    }

    if (rounds <= 0) {
        std::cerr << "Error: rounds must be positive." << std::endl;
        return -1;
    }

    /* Each OpenMP thread gets its own benchmark instances, RNG, and
     * round loop so that threads operate on independent data.              */
    const int num_threads = omp_get_max_threads();
    struct timeval t_wall_start, t_wall_end;

    #pragma omp parallel
    {
        /* Build the list of enabled benchmarks in canonical order.
         * Each entry owns a benchmark instance that is set up once before the
         * round loop and destroyed once after the loop completes.              */
        std::vector<BenchEntry> benches;
        const int tid = omp_get_thread_num();

        if (run_triad) {
            auto* b = new TriadBench;
            b->set_thread_id(tid);
            b->set_bench_name("triad");
            b->set_reps_mult(REPS_MULT_TRIAD);
            b->setup_benchmark(SIZE_TRIAD);
            benches.push_back({"triad", b});
        }
        if (run_cachebwl2) {
            auto* b = new CacheBwBench;
            b->set_thread_id(tid);
            b->set_bench_name("cachebwl2");
            b->set_reps_mult(REPS_MULT_CACHEBWL2);
            b->setup_benchmark(SIZE_CACHEBWL2);
            benches.push_back({"cachebwl2", b});
        }
        if (run_cachebwl3) {
            auto* b = new CacheBwBench;
            b->set_thread_id(tid);
            b->set_bench_name("cachebwl3");
            b->set_reps_mult(REPS_MULT_CACHEBWL3);
            b->setup_benchmark(SIZE_CACHEBWL3);
            benches.push_back({"cachebwl3", b});
        }
        if (run_xsmm) {
            auto* b = new XsmmBench;
            b->set_thread_id(tid);
            b->set_bench_name("xsmm");
            b->set_reps_mult(REPS_MULT_XSMM);
            b->setup_benchmark(0);
            benches.push_back({"xsmm", b});
        }
        if (run_qs) {
            auto* b = new QsBench;
            b->set_thread_id(tid);
            b->set_bench_name("qs");
            b->set_reps_mult(REPS_MULT_QS);
            b->setup_benchmark(SIZE_QS);
            benches.push_back({"qs", b});
        }
        if (run_intipc) {
            auto* b = new IntIpcBench;
            b->set_thread_id(tid);
            b->set_bench_name("intipc");
            b->set_reps_mult(REPS_MULT_INTIPC);
            b->setup_benchmark(SIZE_INTIPC);
            benches.push_back({"intipc", b});
        }
        if (run_latency) {
            auto* b = new LatencyBench;
            b->set_thread_id(tid);
            b->set_bench_name("latency");
            b->set_reps_mult(REPS_MULT_LATENCY);
            b->setup_benchmark(SIZE_LATENCY);
            benches.push_back({"latency", b});
        }
        if (run_sleep) {
            auto* b = new SleepBench;
            b->set_thread_id(tid);
            b->set_bench_name("sleep");
            b->set_reps_mult(REPS_MULT_SLEEP);
            b->setup_benchmark(0);
            benches.push_back({"sleep", b});
        }

        /* Outer rounds loop – each round picks one benchmark at random and
         * a random reps multiplier in [1, RND_REPS], or replays a previously
         * saved schedule.                                                    */
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, benches.size() - 1);
        std::uniform_int_distribution<int> reps_dist(1, RND_REPS);

        /* If replaying, read the schedule (bench index + reps multiplier) from file */
        std::vector<size_t> schedule(rounds);
        std::vector<int>    reps_schedule(rounds);
        if (!replay_prefix.empty()) {
            std::string fname = replay_prefix + "_tid" + std::to_string(tid) + ".sched";
            std::ifstream ifs(fname);
            if (!ifs) {
                #pragma omp critical
                std::cerr << "Error: cannot open schedule file '" << fname << "'" << std::endl;
            } else {
                for (int r = 0; r < rounds; ++r) {
                    ifs >> schedule[r] >> reps_schedule[r];
                }
            }
        } else {
            for (int r = 0; r < rounds; ++r) {
                schedule[r]      = dist(rng);
                reps_schedule[r] = reps_dist(rng);
            }
        }

        /* If dumping, write the schedule (bench index + reps multiplier) to file */
        if (!dump_prefix.empty()) {
            std::string fname = dump_prefix + "_tid" + std::to_string(tid) + ".sched";
            std::ofstream ofs(fname);
            for (int r = 0; r < rounds; ++r) {
                ofs << schedule[r] << " " << reps_schedule[r] << "\n";
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            gettimeofday(&t_wall_start, NULL);
        }

        for (int round = 0; round < rounds; ++round) {
            const auto& e = benches[schedule[round]];
            e.bench->run_benchmark(reps_schedule[round]);
        }

        #pragma omp barrier
        #pragma omp master
        {
            gettimeofday(&t_wall_end, NULL);
        }

        /* Dump merged per-thread timeline to a single CSV file */
        {
            std::vector<const MicroBench*> bench_ptrs;
            for (const auto& e : benches)
                bench_ptrs.push_back(e.bench);
            MicroBench::dump_merged_perf_data(bench_ptrs);
        }

        /* Tear down all benchmarks after the round loop */
        for (auto& e : benches) {
            e.bench->destroy_benchmark();
            delete e.bench;
        }
    } /* end omp parallel */

    const double wall_s = static_cast<double>(
        (t_wall_end.tv_sec  * 1000000 + t_wall_end.tv_usec) -
        (t_wall_start.tv_sec * 1000000 + t_wall_start.tv_usec)) / 1.0e6;
    std::cout << "AgenticCPUBench: " << wall_s << " s, "
              << num_threads << " thread(s), "
              << rounds << " round(s)" << std::endl;

    return 0;
}
