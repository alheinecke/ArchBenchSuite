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

#include "triad_bench.hpp"

#include <cstdlib>
#include <iostream>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE3__)
#include <immintrin.h>
#endif

/* Alignment required by the SIMD kernels (2 MiB huge-page friendly). */
static constexpr size_t ALLOC_ALIGN = 2097152;

/* Round element count down to a multiple of this so that every SIMD path
 * (AVX-512 steps by 8 doubles) sees an aligned trip count.                */
static constexpr size_t ELEM_ALIGN  = 8;

/* -------------------------------------------------------------------------
 * Platform-specific triad kernel: A[i] = B[i] + scalar * C[i]
 * Adopted from level0/stream/stream.c intrinsic implementations.
 * -------------------------------------------------------------------------*/
static void kernel_triad(double* a, const double* b, const double* c,
                         double scalar, size_t n) {
#if defined(__AVX512F__)
    __m512d vecscalar = _mm512_set1_pd(scalar);
    for (size_t j = 0; j < n; j += 8) {
        _mm_prefetch((const char *)&b[j + 64], _MM_HINT_T2);
        _mm_prefetch((const char *)&c[j + 64], _MM_HINT_T2);
        _mm_prefetch((const char *)&b[j + 16], _MM_HINT_T1);
        _mm_prefetch((const char *)&c[j + 16], _MM_HINT_T1);
        _mm512_stream_pd(
            &a[j],
            _mm512_add_pd(_mm512_load_pd(&b[j]),
                          _mm512_mul_pd(vecscalar, _mm512_load_pd(&c[j]))));
    }
#elif defined(__AVX2__) || defined(__AVX__)
    __m256d vecscalar = _mm256_set1_pd(scalar);
    for (size_t j = 0; j < n; j += 4)
        _mm256_stream_pd(
            &a[j],
            _mm256_add_pd(_mm256_load_pd(&b[j]),
                          _mm256_mul_pd(vecscalar, _mm256_load_pd(&c[j]))));
#elif defined(__SSE3__)
    __m128d vecscalar = _mm_set1_pd(scalar);
    for (size_t j = 0; j < n; j += 2)
        _mm_stream_pd(&a[j],
                      _mm_add_pd(_mm_load_pd(&b[j]),
                                 _mm_mul_pd(vecscalar, _mm_load_pd(&c[j]))));
#elif defined(__aarch64__) || defined(__ARM_NEON)
    __asm__ __volatile__(
        "mov x0, %0\n\t"
        "mov x1, %1\n\t"
        "mov x2, %2\n\t"
        "mov x3, %3\n\t"
        "mov x4, %4\n\t"
        "ldr d6, [x3]\n\t"
        "1:\n\t"
        "ldr  d0, [x1]\n\t"
        "ldr  d1, [x2]\n\t"
        "fmadd  d0, d6, d1, d0\n\t"
        "ldr  d2, [x1,8]\n\t"
        "ldr  d3, [x2,8]\n\t"
        "fmadd  d2, d6, d3, d2\n\t"
        "stnp d0, d2, [x0]\n\t"
        "ldr  d4, [x1,16]\n\t"
        "ldr  d5, [x2,16]\n\t"
        "fmadd  d4, d6, d5, d4\n\t"
        "ldr  d7, [x1,24]\n\t"
        "ldr  d8, [x2,24]\n\t"
        "fmadd  d7, d6, d8, d7\n\t"
        "stnp d4, d7, [x0,16]\n\t"
        "add x0, x0, #32\n\t"
        "add x1, x1, #32\n\t"
        "add x2, x2, #32\n\t"
        "sub x4, x4, #4\n\t"
        "cbnz x4, 1b\n\t"
        :
        : "r"(a), "r"(b), "r"(c), "r"(&scalar), "r"(n)
        : "x0", "x1", "x2", "x3", "x4",
          "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8");
#else
    for (size_t j = 0; j < n; ++j)
        a[j] = b[j] + scalar * c[j];
#endif
}

/* -------------------------------------------------------------------------
 * TriadBench implementation
 * -------------------------------------------------------------------------*/

TriadBench::TriadBench() = default;

TriadBench::~TriadBench() {
    destroy_benchmark();
}

void TriadBench::setup_benchmark(size_t size_bytes) {
    size_bytes_ = size_bytes;
    const size_t n_raw = size_bytes / sizeof(double);
    n_ = (n_raw / ELEM_ALIGN) * ELEM_ALIGN;

    A_ = static_cast<double*>(aligned_alloc(ALLOC_ALIGN, n_ * sizeof(double)));
    B_ = static_cast<double*>(aligned_alloc(ALLOC_ALIGN, n_ * sizeof(double)));
    C_ = static_cast<double*>(aligned_alloc(ALLOC_ALIGN, n_ * sizeof(double)));

    if (!A_ || !B_ || !C_) {
        std::cerr << "TriadBench::setup_benchmark: memory allocation failed." << std::endl;
        destroy_benchmark();
        return;
    }

    for (size_t i = 0; i < n_; ++i) {
        A_[i] = 0.0;
        B_[i] = static_cast<double>(i);
        C_[i] = static_cast<double>(n_ - i);
    }

    /* warm-up: two passes to bring data into caches / TLBs */
    kernel_triad(A_, B_, C_, scalar_, n_);
    kernel_triad(A_, B_, C_, scalar_, n_);
}

void TriadBench::run_benchmark(int reps) {
    reps *= reps_mult_;
    struct timeval t_start, t_end;

    const double mono_ts = capture_monotonic();

    gettimeofday(&t_start, NULL);
    for (int r = 0; r < reps; ++r) {
        kernel_triad(A_, B_, C_, scalar_, n_);
    }
    gettimeofday(&t_end, NULL);

    const double t = sec(t_start, t_end);

    /* bytes moved: 2 reads (B, C) + 1 write (A) per rep */
    const double bytes_per_rep = 3.0 * static_cast<double>(n_) * sizeof(double);
    /* FP ops: n multiplications + n additions per rep */
    const double flops_per_rep = 2.0 * static_cast<double>(n_);
    const double bw_gbs = (bytes_per_rep * reps / 1.0e9) / t;
    const double gflops = (flops_per_rep * reps / 1.0e9) / t;
    const double ai     = flops_per_rep / bytes_per_rep;

    PerfSample s;
    s.bench_name       = bench_name_;
    s.reps             = reps;
    s.monotonic_time_s = mono_ts;
    s.elapsed_s        = t;
    s.metrics.push_back({"array_size_MB", size_bytes_ / (1024.0*1024.0)});
    s.metrics.push_back({"n_elements", static_cast<double>(n_)});
    s.metrics.push_back({"bandwidth_GBs", bw_gbs});
    s.metrics.push_back({"flops", flops_per_rep * reps});
    s.metrics.push_back({"GFLOPS", gflops});
    s.metrics.push_back({"arithmetic_intensity_FLOP_byte", ai});
    perf_data_.push_back(s);
}

void TriadBench::destroy_benchmark() {
    free(A_); A_ = nullptr;
    free(B_); B_ = nullptr;
    free(C_); C_ = nullptr;
    n_ = 0;
}
