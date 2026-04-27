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

#ifndef LATENCY_BENCH_HPP
#define LATENCY_BENCH_HPP

#include "microbench.hpp"
#include <cstddef>
#include <cstdint>

/**
 * LatencyBench – idle memory-latency benchmark modelled on the
 * HPCC RandomAccess / GUPS specification.
 *
 * Kernel (per repetition):
 *   uint64_t ran = 1;   // deterministic seed – same every rep
 *   for (uint64_t u = 0; u < num_updates; ++u) {
 *       ran       = (ran << 1) ^ (((int64_t)ran < 0) ? POLY : 0);
 *       table[ran & mask] ^= ran;
 *   }
 *
 * The PRNG is the HPCC-specified 64-bit primitive-polynomial LFSR:
 *   POLY = 0x0000000000000007ULL
 *
 * The table is sized to the nearest power-of-two at or below
 *   size_bytes / sizeof(uint64_t)
 * so that index masking (ran & mask) replaces modulo.
 * For large tables (> last-level cache) each XOR update generates a
 * random cache miss, stressing memory-subsystem latency.
 *
 * Number of updates per repetition: 4 * n  (following the HPCC spec).
 *
 * Reported metric: GUPS (10^9 updates / second).
 *   GUPS = num_updates / elapsed / 1e9
 */
class LatencyBench : public MicroBench {
public:
    LatencyBench();
    ~LatencyBench() override;

    void setup_benchmark(size_t size_bytes) override;
    void run_benchmark(int reps) override;
    void destroy_benchmark() override;

    /** Number of uint64_t elements in the table (set after setup_benchmark). */
    size_t get_n() const { return n_; }

private:
    void run_kernel() const;

    /* HPCC LFSR polynomial */
    static constexpr uint64_t POLY = 0x0000000000000007ULL;

    size_t    size_bytes_  = 0;
    size_t    n_           = 0;   /**< Table length (power of two).          */
    uint64_t  mask_        = 0;   /**< n_ - 1, used for index masking.       */
    uint64_t  num_updates_ = 0;   /**< 4 * n_ following the HPCC spec.       */
    uint64_t* table_       = nullptr;
};

#endif /* LATENCY_BENCH_HPP */
