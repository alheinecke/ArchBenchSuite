/******************************************************************************
** Copyright (c) 2026, Alexander Heinecke                                   **
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

#ifndef CACHEBW_BENCH_HPP
#define CACHEBW_BENCH_HPP

#include "microbench.hpp"
#include <cstddef>

/**
 * CacheBwBench – pure sequential read-bandwidth benchmark.
 *
 * Kernel: reads an array of doubles with platform-specific SIMD loads.
 *   - 1 load per element => n * sizeof(double) bytes per pass
 *   - 0 FP ops
 *   => arithmetic intensity = 0 FLOP/byte (pure memory benchmark)
 *
 * The working-set size is rounded down to a multiple of 256 doubles so
 * that the platform-specific assembly kernels always see an aligned count.
 */
class CacheBwBench : public MicroBench {
public:
    CacheBwBench();
    ~CacheBwBench() override;

    void setup_benchmark(size_t size_bytes) override;
    void run_benchmark(int reps) override;
    void destroy_benchmark() override;

    /** Number of double elements in the array (set after setup_benchmark). */
    size_t get_n() const { return n_; }

private:
    /** Execute one timed pass of the read kernel over data_. */
    void run_kernel() const;

    size_t  n_          = 0;
    size_t  size_bytes_ = 0;
    double* data_       = nullptr;
};

#endif /* CACHEBW_BENCH_HPP */
