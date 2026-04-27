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

#ifndef INTIPC_BENCH_HPP
#define INTIPC_BENCH_HPP

#include "microbench.hpp"
#include <cstddef>
#include <cstdint>

/**
 * IntIpcBench – integer IPC stress benchmark.
 *
 * Kernel (conceptual C):
 *   int64_t data[M][N];   // M=8 rows, N=4096 columns, fixed size
 *   int64_t accum[M];
 *
 *   for (iter = 0; iter < num_iter; ++iter)
 *     for (i = 0; i < N; ++i)
 *       for (j = 0; j < M; ++j)   // fully unrolled in asm
 *         accum[j] += data[j][i];
 *
 * The j-loop (M=8) is fully unrolled in platform inline assembly using only
 * scalar integer registers (no SSE/AVX/NEON/SVE).  The i-loop body is further
 * unrolled 4× (32 load+add pairs per iteration) to fill the decode window.
 *
 * The data array is small (8 × 4096 × 8 = 256 KiB) and fits in L2 cache.
 * num_iter = size_mb * NUM_ITER_SCALE so that wall-clock time scales with the
 * size_mb argument.
 *
 * Reported metric: GOPS (10^9 integer add operations per second).
 *   ops_per_iter = M * N = 8 * 4096 = 32768
 *   total_ops    = num_iter * ops_per_iter
 */
class IntIpcBench : public MicroBench {
public:
    IntIpcBench();
    ~IntIpcBench() override;

    void setup_benchmark(size_t size_bytes) override;
    void run_benchmark(int reps) override;
    void destroy_benchmark() override;

private:
    void run_kernel(int64_t num_iter) const;

    /* M=8 rows, N=4096 columns, row-major: data_[j][i] */
    static constexpr int64_t M = 8;
    static constexpr int64_t N = 4096;
    /* Each MiB of working set maps to this many kernel iterations */
    static constexpr int64_t NUM_ITER_SCALE = 1024;

    size_t   size_bytes_ = 0;
    int64_t  num_iter_ = 0;
    int64_t* data_     = nullptr;  /* [M * N] elements, row-major */
    int64_t  accum_[M] = {};       /* accumulator sinks (prevent DCE) */
};

#endif /* INTIPC_BENCH_HPP */
