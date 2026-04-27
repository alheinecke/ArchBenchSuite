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

#ifndef QS_BENCH_HPP
#define QS_BENCH_HPP

#include "microbench.hpp"
#include <cstddef>
#include <cstdint>

/**
 * QsBench – quicksort benchmark on an array of int64_t values.
 *
 * Kernel: in-place quicksort with median-of-3 pivot selection and
 *         insertion sort for sub-arrays smaller than 16 elements.
 *
 * Each repetition:
 *   1. Copies the fixed, pre-shuffled reference array into the work buffer.
 *   2. Times a single full quicksort pass over the work buffer.
 *
 * The copy is intentionally outside the timed region so that each rep
 * benchmarks the sort itself on an unsorted (random) dataset.
 *
 * Reported metric: throughput in million elements per second (Melements/s).
 */
class QsBench : public MicroBench {
public:
    QsBench();
    ~QsBench() override;

    void setup_benchmark(size_t size_bytes) override;
    void run_benchmark(int reps) override;
    void destroy_benchmark() override;

    /** Number of int64_t elements in the array (set after setup_benchmark). */
    size_t get_n() const { return n_; }

private:
    size_t   n_          = 0;
    size_t   size_bytes_ = 0;
    int64_t* data_    = nullptr;  /**< Work buffer – sorted in-place each rep. */
    int64_t* ref_     = nullptr;  /**< Fixed shuffled reference, never modified. */
};

#endif /* QS_BENCH_HPP */
