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

#include "xsmm_bench.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>

/* -------------------------------------------------------------------------
 * Fixed GEMM parameters
 * Column-major layout:
 *   A : M×K  =>  lda = M
 *   B : K×N  =>  ldb = K
 *   C : M×N  =>  ldc = M
 * -------------------------------------------------------------------------*/
static constexpr libxsmm_blasint XSMM_M        = 64;
static constexpr libxsmm_blasint XSMM_N        = 24;
static constexpr libxsmm_blasint XSMM_K        = 64;
static constexpr libxsmm_blasint XSMM_LDA      = XSMM_M;
static constexpr libxsmm_blasint XSMM_LDB      = XSMM_K;
static constexpr libxsmm_blasint XSMM_LDC      = XSMM_M;
static constexpr libxsmm_blasint XSMM_BR_COUNT = 16;

/* 64-byte alignment for SIMD loads */
static constexpr size_t XSMM_ALLOC_ALIGN = 64;

/* FLOPS per single kernel invocation: 2 * M * N * K * br_count */
static constexpr double XSMM_FLOPS_PER_CALL =
    2.0 * XSMM_M * XSMM_N * XSMM_K * XSMM_BR_COUNT;

/* -------------------------------------------------------------------------
 * XsmmBench implementation
 * -------------------------------------------------------------------------*/

XsmmBench::XsmmBench() = default;

XsmmBench::~XsmmBench() {
    destroy_benchmark();
}

void XsmmBench::setup_benchmark(size_t size_bytes) {
    size_bytes_ = size_bytes; /* stored; does not affect kernel sizing */

    /* Allocate matrix buffers.
     * A: br_count batches of LDA×K floats (col-major)
     * B: br_count batches of LDB×N floats (col-major)
     * C: one LDC×N float matrix (beta=0: always overwritten)            */
    const size_t a_elems = static_cast<size_t>(XSMM_BR_COUNT) * XSMM_LDA * XSMM_K;
    const size_t b_elems = static_cast<size_t>(XSMM_BR_COUNT) * XSMM_LDB * XSMM_N;
    const size_t c_elems = static_cast<size_t>(XSMM_LDC) * XSMM_N;

    A_ = static_cast<float*>(aligned_alloc(XSMM_ALLOC_ALIGN, a_elems * sizeof(float)));
    B_ = static_cast<float*>(aligned_alloc(XSMM_ALLOC_ALIGN, b_elems * sizeof(float)));
    C_ = static_cast<float*>(aligned_alloc(XSMM_ALLOC_ALIGN, c_elems * sizeof(float)));

    if (!A_ || !B_ || !C_) {
        std::cerr << "XsmmBench::setup_benchmark: memory allocation failed." << std::endl;
        destroy_benchmark();
        return;
    }

    /* Initialise A and B with small random values, zero C */
    srand(42);
    for (size_t i = 0; i < a_elems; ++i)
        A_[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
    for (size_t i = 0; i < b_elems; ++i)
        B_[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
    for (size_t i = 0; i < c_elems; ++i)
        C_[i] = 0.0f;

    /* -----------------------------------------------------------------------
     * Dispatch the JIT kernel
     * flags  : NN layout, beta=0
     * br type: STRIDE – strides are baked into the JIT'd code
     * ----------------------------------------------------------------------- */
    const libxsmm_bitfield flags =
        LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_BETA_0;

    const libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(
        XSMM_M, XSMM_N, XSMM_K,
        XSMM_LDA, XSMM_LDB, XSMM_LDC,
        LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
        LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);

    libxsmm_gemm_batch_reduce_config brconfig;
    brconfig.br_type          = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    /* strides in bytes between consecutive A / B matrices */
    brconfig.br_stride_a_hint =
        static_cast<libxsmm_blasint>(XSMM_LDA * XSMM_K * sizeof(float));
    brconfig.br_stride_b_hint =
        static_cast<libxsmm_blasint>(XSMM_LDB * XSMM_N * sizeof(float));
    /* fully unroll the BR loop – br_count is known at JIT time */
    brconfig.br_unroll_hint   = static_cast<unsigned char>(XSMM_BR_COUNT);

    kernel_ = libxsmm_dispatch_brgemm(shape, flags, /*prefetch=*/0, brconfig);

    if (kernel_ == nullptr) {
        std::cerr << "XsmmBench::setup_benchmark: libxsmm JIT dispatch failed. "
                     "Run with LIBXSMM_VERBOSE=-1 for details." << std::endl;
        return;
    }

    br_count_ = static_cast<unsigned long long>(XSMM_BR_COUNT);

    /* Warm-up: two passes to prime i-cache and branch predictors */
    libxsmm_gemm_param param;
    memset(&param, 0, sizeof(param));
    param.a.primary   = static_cast<void*>(A_);
    param.b.primary   = static_cast<void*>(B_);
    param.c.primary   = static_cast<void*>(C_);
    param.op.tertiary = static_cast<void*>(&br_count_);
    kernel_(&param);
    kernel_(&param);
}

void XsmmBench::run_benchmark(int reps) {
    reps *= reps_mult_;
    if (kernel_ == nullptr) {
        std::cerr << "XsmmBench::run_benchmark: kernel not initialised." << std::endl;
        return;
    }

    struct timeval t_start, t_end;

    /* Prepare the param struct once – a/b/c pointers do not change */
    libxsmm_gemm_param param;
    memset(&param, 0, sizeof(param));
    param.a.primary   = static_cast<void*>(A_);
    param.b.primary   = static_cast<void*>(B_);
    param.c.primary   = static_cast<void*>(C_);
    param.op.tertiary = static_cast<void*>(&br_count_);

    const double mono_ts = capture_monotonic();

    gettimeofday(&t_start, NULL);
    for (int r = 0; r < reps; ++r) {
        kernel_(&param);
    }
    gettimeofday(&t_end, NULL);

    const double t = sec(t_start, t_end);

    const double gflops = (XSMM_FLOPS_PER_CALL * reps / 1.0e9) / t;

    PerfSample s;
    s.bench_name       = bench_name_;
    s.reps             = reps;
    s.monotonic_time_s = mono_ts;
    s.elapsed_s        = t;
    s.metrics.push_back({"M", static_cast<double>(XSMM_M)});
    s.metrics.push_back({"N", static_cast<double>(XSMM_N)});
    s.metrics.push_back({"K", static_cast<double>(XSMM_K)});
    s.metrics.push_back({"br_count", static_cast<double>(XSMM_BR_COUNT)});
    s.metrics.push_back({"GFLOPS", gflops});
    perf_data_.push_back(s);
}

void XsmmBench::destroy_benchmark() {
    free(A_); A_ = nullptr;
    free(B_); B_ = nullptr;
    free(C_); C_ = nullptr;
    kernel_   = nullptr;
    br_count_ = 0;
}
