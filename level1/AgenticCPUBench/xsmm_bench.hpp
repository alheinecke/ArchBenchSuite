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

#ifndef XSMM_BENCH_HPP
#define XSMM_BENCH_HPP

#include "microbench.hpp"
#include <cstddef>
#include <libxsmm.h>

/**
 * XsmmBench – F32 strided-BRGEMM benchmark using libxsmm JIT.
 *
 * Fixed kernel parameters:
 *   M=64, N=24, K=64, LDA=64, LDB=64, LDC=64
 *   alpha=1, beta=0
 *   BRGEMM mode: strided (LIBXSMM_GEMM_BATCH_REDUCE_STRIDE)
 *   br_count=16
 *
 * FLOPS per kernel call: 2 * M * N * K * br_count
 *                       = 2 * 64 * 24 * 64 * 16 = 3,145,728
 */
class XsmmBench : public MicroBench {
public:
    XsmmBench();
    ~XsmmBench() override;

    void setup_benchmark(size_t size_bytes) override;
    void run_benchmark(int reps) override;
    void destroy_benchmark() override;

private:
    libxsmm_gemmfunction kernel_      = nullptr;
    unsigned long long   br_count_    = 0;
    size_t               size_bytes_  = 0;
    float*               A_        = nullptr;
    float*               B_        = nullptr;
    float*               C_        = nullptr;
};

#endif /* XSMM_BENCH_HPP */
