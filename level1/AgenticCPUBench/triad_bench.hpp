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

#ifndef TRIAD_BENCH_HPP
#define TRIAD_BENCH_HPP

#include "microbench.hpp"
#include <cstddef>

/**
 * TriadBench – memory-bandwidth / FP-compute benchmark.
 *
 * Kernel: A[i] = B[i] + scalar * C[i]
 *   - 2 loads + 1 store  =>  3 * sizeof(double) bytes per element
 *   - 2 FP ops (mul + add) per element
 *   => arithmetic intensity ~ 2 / (3*8) ~ 0.083 FLOP/byte
 */
class TriadBench : public MicroBench {
public:
    TriadBench();
    ~TriadBench() override;

    void setup_benchmark(size_t size_bytes) override;
    void run_benchmark(int reps) override;
    void destroy_benchmark() override;

    /** Number of double elements in each array (set after setup_benchmark). */
    size_t get_n() const { return n_; }

private:
    size_t  n_          = 0;
    size_t  size_bytes_ = 0;
    double* A_       = nullptr;
    double* B_       = nullptr;
    double* C_       = nullptr;
    double  scalar_  = 3.0;
};

#endif /* TRIAD_BENCH_HPP */
