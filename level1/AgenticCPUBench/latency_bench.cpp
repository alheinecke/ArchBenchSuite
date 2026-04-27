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

#include "latency_bench.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>

/* -------------------------------------------------------------------------
 * Largest power of two that is <= x.
 * -------------------------------------------------------------------------*/
static size_t floor_pow2(size_t x) {
    if (x == 0) return 0;
    size_t p = 1;
    while (p * 2 <= x) p *= 2;
    return p;
}

/* =========================================================================
 * GUPS / RandomAccess kernel
 *
 * Implements the HPCC RandomAccess specification:
 *   - LFSR with primitive polynomial over GF(2): x^64 + x^4 + x^3 + x + 1
 *     feedback polynomial: POLY = 0x0000000000000007
 *     ran = (ran << 1) ^ (ran_msb_set ? POLY : 0)
 *   - Table update: table[ran & mask] ^= ran
 *   - num_updates = 4 * n  (HPCC requirement)
 *   - Seed is fixed at 1 every call so each rep is identical and
 *     results are reproducible.
 *
 * Why this stresses memory latency:
 *   The LFSR output covers the full 2^64 period; for a large table the
 *   index sequence is pseudo-random with no temporal locality.  Almost
 *   every table access generates a last-level-cache miss, forcing the
 *   processor to wait for DRAM.  Because each XOR update is a
 *   read-modify-write, the load and store share the same cache line,
 *   so the effective latency per update is one DRAM round-trip.
 * =========================================================================*/
void LatencyBench::run_kernel() const {
    uint64_t* __restrict__ tbl  = table_;
    const uint64_t         msk  = mask_;
    const uint64_t         nupd = num_updates_;
    const uint64_t         poly = POLY;

    uint64_t ran = 1ULL;   /* deterministic seed */

    for (uint64_t u = 0; u < nupd; ++u) {
        /* HPCC LFSR: shift left, XOR polynomial if MSB was set */
        ran = (ran << 1) ^ (static_cast<int64_t>(ran) < 0 ? poly : 0ULL);
        tbl[ran & msk] ^= ran;
    }
}

/* =========================================================================
 * LatencyBench public interface
 * =========================================================================*/

LatencyBench::LatencyBench() = default;

LatencyBench::~LatencyBench() {
    destroy_benchmark();
}

void LatencyBench::setup_benchmark(size_t size_bytes) {
    size_bytes_ = size_bytes;

    /* Round table length down to a power of two so masking works correctly */
    const size_t raw_n = size_bytes / sizeof(uint64_t);
    n_    = floor_pow2(raw_n);
    mask_ = static_cast<uint64_t>(n_) - 1ULL;

    if (n_ == 0) {
        std::cerr << "LatencyBench::setup_benchmark: size_bytes too small "
                     "(need at least 8 bytes for one element)." << std::endl;
        return;
    }

    /* HPCC: number of updates = 4 * table_length */
    num_updates_ = 4ULL * static_cast<uint64_t>(n_);

    table_ = static_cast<uint64_t*>(
        aligned_alloc(64, n_ * sizeof(uint64_t)));

    if (!table_) {
        std::cerr << "LatencyBench::setup_benchmark: memory allocation failed."
                  << std::endl;
        n_ = 0;
        return;
    }

    /* Initialise table following the HPCC spec: table[i] = i */
    for (size_t i = 0; i < n_; ++i)
        table_[i] = static_cast<uint64_t>(i);

    /* Warm-up pass to bring the table into whatever cache level it fits */
    run_kernel();
    /* Re-initialise so the first timed rep starts from a known state */
    for (size_t i = 0; i < n_; ++i)
        table_[i] = static_cast<uint64_t>(i);
}

void LatencyBench::run_benchmark(int reps) {
    reps *= reps_mult_;
    if (!table_) {
        std::cerr << "LatencyBench::run_benchmark: benchmark not initialised."
                  << std::endl;
        return;
    }

    struct timeval t_start, t_end;

    const double mono_ts = capture_monotonic();

    gettimeofday(&t_start, NULL);
    for (int r = 0; r < reps; ++r) {
        /* Re-initialise the table before each rep so every rep performs
         * the same work and results are comparable across runs.            */
        for (size_t i = 0; i < n_; ++i)
            table_[i] = static_cast<uint64_t>(i);

        run_kernel();
    }
    gettimeofday(&t_end, NULL);

    const double t = sec(t_start, t_end);

    const double gups = static_cast<double>(num_updates_) * reps / t / 1.0e9;
    /* Actual table size in MiB (power-of-two, may differ from size_bytes_) */
    const double actual_mb = static_cast<double>(n_) * sizeof(uint64_t)
                             / (1024.0 * 1024.0);

    PerfSample s;
    s.bench_name       = bench_name_;
    s.reps             = reps;
    s.monotonic_time_s = mono_ts;
    s.elapsed_s        = t;
    s.metrics.push_back({"table_size_MB", actual_mb});
    s.metrics.push_back({"n_elements", static_cast<double>(n_)});
    s.metrics.push_back({"num_updates", static_cast<double>(num_updates_)});
    s.metrics.push_back({"GUPS", gups});
    perf_data_.push_back(s);
}

void LatencyBench::destroy_benchmark() {
    free(table_);
    table_       = nullptr;
    n_           = 0;
    mask_        = 0;
    num_updates_ = 0;
    size_bytes_  = 0;
}
