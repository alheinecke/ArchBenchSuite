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

#include "cachebw_bench.hpp"

#include <cstdlib>
#include <iostream>

/* Alignment required by the SIMD kernels (2 MiB huge-page friendly). */
static constexpr size_t ALLOC_ALIGN = 2097152;

/* The SIMD kernels count elements in chunks of 256 doubles; round down. */
static constexpr size_t ELEM_ALIGN  = 256;

/* -------------------------------------------------------------------------
 * Platform-specific read kernels.
 * Each variant reads the whole array sequentially using SIMD loads.
 * The element count (sz) is passed by pointer to match the original
 * cachebw.c pattern; the asm reads it from memory with a single movq.
 * -------------------------------------------------------------------------*/
static void kernel_read(double* ptr, const size_t* psz) {
#if defined(__AVX512F__)
    __asm__ __volatile__(
        "movq  %0,       %%r8\n\t"
        "movq  %1,       %%r10\n\t"
        "movq  (%%r10),  %%r9\n\t"
        "1:\n\t"
        "subq  $256,     %%r9\n\t"
        "vmovapd     0(%%r8),  %%zmm0\n\t"
        "vmovapd    64(%%r8),  %%zmm1\n\t"
        "vmovapd   128(%%r8),  %%zmm2\n\t"
        "vmovapd   192(%%r8),  %%zmm3\n\t"
        "vmovapd   256(%%r8),  %%zmm4\n\t"
        "vmovapd   320(%%r8),  %%zmm5\n\t"
        "vmovapd   384(%%r8),  %%zmm6\n\t"
        "vmovapd   448(%%r8),  %%zmm7\n\t"
        "vmovapd   512(%%r8),  %%zmm8\n\t"
        "vmovapd   576(%%r8),  %%zmm9\n\t"
        "vmovapd   640(%%r8), %%zmm10\n\t"
        "vmovapd   704(%%r8), %%zmm11\n\t"
        "vmovapd   768(%%r8), %%zmm12\n\t"
        "vmovapd   832(%%r8), %%zmm13\n\t"
        "vmovapd   896(%%r8), %%zmm14\n\t"
        "vmovapd   960(%%r8), %%zmm15\n\t"
        "vmovapd  1024(%%r8), %%zmm16\n\t"
        "vmovapd  1088(%%r8), %%zmm17\n\t"
        "vmovapd  1152(%%r8), %%zmm18\n\t"
        "vmovapd  1216(%%r8), %%zmm19\n\t"
        "vmovapd  1280(%%r8), %%zmm20\n\t"
        "vmovapd  1344(%%r8), %%zmm21\n\t"
        "vmovapd  1408(%%r8), %%zmm22\n\t"
        "vmovapd  1472(%%r8), %%zmm23\n\t"
        "vmovapd  1536(%%r8), %%zmm24\n\t"
        "vmovapd  1600(%%r8), %%zmm25\n\t"
        "vmovapd  1664(%%r8), %%zmm26\n\t"
        "vmovapd  1728(%%r8), %%zmm27\n\t"
        "vmovapd  1792(%%r8), %%zmm28\n\t"
        "vmovapd  1856(%%r8), %%zmm29\n\t"
        "vmovapd  1920(%%r8), %%zmm30\n\t"
        "vmovapd  1984(%%r8), %%zmm31\n\t"
        "addq  $2048, %%r8\n\t"
        "cmpq  $0,    %%r9\n\t"
        "jg   1b\n\t"
        : : "m"(ptr), "m"(psz)
        : "r8","r9","r10",
          "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7",
          "xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15",
          "xmm16","xmm17","xmm18","xmm19","xmm20","xmm21","xmm22","xmm23",
          "xmm24","xmm25","xmm26","xmm27","xmm28","xmm29","xmm30","xmm31");
#elif defined(__AVX__)
    __asm__ __volatile__(
        "movq  %0,       %%r8\n\t"
        "movq  %1,       %%r10\n\t"
        "movq  (%%r10),  %%r9\n\t"
        "1:\n\t"
        "subq  $64,     %%r9\n\t"
        "vmovapd    0(%%r8),  %%ymm0\n\t"
        "vmovapd   32(%%r8),  %%ymm1\n\t"
        "vmovapd   64(%%r8),  %%ymm2\n\t"
        "vmovapd   96(%%r8),  %%ymm3\n\t"
        "vmovapd  128(%%r8),  %%ymm4\n\t"
        "vmovapd  160(%%r8),  %%ymm5\n\t"
        "vmovapd  192(%%r8),  %%ymm6\n\t"
        "vmovapd  224(%%r8),  %%ymm7\n\t"
        "vmovapd  256(%%r8),  %%ymm8\n\t"
        "vmovapd  288(%%r8),  %%ymm9\n\t"
        "vmovapd  320(%%r8), %%ymm10\n\t"
        "vmovapd  352(%%r8), %%ymm11\n\t"
        "vmovapd  384(%%r8), %%ymm12\n\t"
        "vmovapd  416(%%r8), %%ymm13\n\t"
        "vmovapd  448(%%r8), %%ymm14\n\t"
        "vmovapd  480(%%r8), %%ymm15\n\t"
        "addq  $512,  %%r8\n\t"
        "cmpq  $0,    %%r9\n\t"
        "jg   1b\n\t"
        : : "m"(ptr), "m"(psz)
        : "r8","r9","r10",
          "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7",
          "xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
#elif defined(__SSE3__)
    __asm__ __volatile__(
        "movq  %0,       %%r8\n\t"
        "movq  %1,       %%r10\n\t"
        "movq  (%%r10),  %%r9\n\t"
        "1:\n\t"
        "subq  $32,     %%r9\n\t"
        "movapd    0(%%r8),  %%xmm0\n\t"
        "movapd   16(%%r8),  %%xmm1\n\t"
        "movapd   32(%%r8),  %%xmm2\n\t"
        "movapd   48(%%r8),  %%xmm3\n\t"
        "movapd   64(%%r8),  %%xmm4\n\t"
        "movapd   80(%%r8),  %%xmm5\n\t"
        "movapd   96(%%r8),  %%xmm6\n\t"
        "movapd  112(%%r8),  %%xmm7\n\t"
        "movapd  128(%%r8),  %%xmm8\n\t"
        "movapd  144(%%r8),  %%xmm9\n\t"
        "movapd  160(%%r8), %%xmm10\n\t"
        "movapd  176(%%r8), %%xmm11\n\t"
        "movapd  192(%%r8), %%xmm12\n\t"
        "movapd  208(%%r8), %%xmm13\n\t"
        "movapd  224(%%r8), %%xmm14\n\t"
        "movapd  240(%%r8), %%xmm15\n\t"
        "addq  $256,  %%r8\n\t"
        "cmpq  $0,    %%r9\n\t"
        "jg   1b\n\t"
        : : "m"(ptr), "m"(psz)
        : "r8","r9","r10",
          "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7",
          "xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
#elif defined(__ARM_NEON)
    size_t* l_parraySize = psz;
    double* l_locAddr    = ptr;
    __asm__ __volatile__(
        "mov x0, %0\n\t"
        "mov x1, %1\n\t"
        "1:\n\t"
        "ld1  {v0.2d},  [x0],16\n\t"
        "ld1  {v1.2d},  [x0],16\n\t"
        "ld1  {v2.2d},  [x0],16\n\t"
        "ld1  {v3.2d},  [x0],16\n\t"
        "ld1  {v4.2d},  [x0],16\n\t"
        "ld1  {v5.2d},  [x0],16\n\t"
        "ld1  {v6.2d},  [x0],16\n\t"
        "ld1  {v7.2d},  [x0],16\n\t"
        "ld1  {v8.2d},  [x0],16\n\t"
        "ld1  {v9.2d},  [x0],16\n\t"
        "ld1 {v10.2d},  [x0],16\n\t"
        "ld1 {v11.2d},  [x0],16\n\t"
        "ld1 {v12.2d},  [x0],16\n\t"
        "ld1 {v13.2d},  [x0],16\n\t"
        "ld1 {v14.2d},  [x0],16\n\t"
        "ld1 {v15.2d},  [x0],16\n\t"
        "ld1 {v16.2d},  [x0],16\n\t"
        "ld1 {v17.2d},  [x0],16\n\t"
        "ld1 {v18.2d},  [x0],16\n\t"
        "ld1 {v19.2d},  [x0],16\n\t"
        "ld1 {v20.2d},  [x0],16\n\t"
        "ld1 {v21.2d},  [x0],16\n\t"
        "ld1 {v22.2d},  [x0],16\n\t"
        "ld1 {v23.2d},  [x0],16\n\t"
        "ld1 {v24.2d},  [x0],16\n\t"
        "ld1 {v25.2d},  [x0],16\n\t"
        "ld1 {v26.2d},  [x0],16\n\t"
        "ld1 {v27.2d},  [x0],16\n\t"
        "ld1 {v28.2d},  [x0],16\n\t"
        "ld1 {v29.2d},  [x0],16\n\t"
        "ld1 {v30.2d},  [x0],16\n\t"
        "ld1 {v31.2d},  [x0],16\n\t"
        "sub x1, x1, #64\n\t"
        "cbnz x1, 1b\n\t"
        : : "r"(l_locAddr), "r"(l_parraySize)
        : "x0","x1",
          "v0","v1","v2","v3","v4","v5","v6","v7",
          "v8","v9","v10","v11","v12","v13","v14","v15",
          "v16","v17","v18","v19","v20","v21","v22","v23",
          "v24","v25","v26","v27","v28","v29","v30","v31");
#else
    /* Generic scalar fallback: accumulate into a volatile sink so the
     * compiler cannot eliminate the reads. */
    volatile double sink = 0.0;
    const size_t n = *psz;
    for (size_t i = 0; i < n; ++i)
        sink += ptr[i];
    (void)sink;
#endif
}

/* -------------------------------------------------------------------------
 * CacheBwBench implementation
 * -------------------------------------------------------------------------*/

CacheBwBench::CacheBwBench() = default;

CacheBwBench::~CacheBwBench() {
    destroy_benchmark();
}

void CacheBwBench::setup_benchmark(size_t size_bytes) {
    size_bytes_ = size_bytes;
    const size_t n_raw = size_bytes / sizeof(double);
    /* Round down to a multiple of ELEM_ALIGN so the asm loop counter stays
     * non-negative throughout execution. */
    n_ = (n_raw / ELEM_ALIGN) * ELEM_ALIGN;

    data_ = static_cast<double*>(aligned_alloc(ALLOC_ALIGN, n_ * sizeof(double)));
    if (!data_) {
        std::cerr << "CacheBwBench::setup_benchmark: memory allocation failed." << std::endl;
        return;
    }

    for (size_t i = 0; i < n_; ++i)
        data_[i] = static_cast<double>(i);

    /* warm-up: two passes to bring data into the target cache level */
    run_kernel();
    run_kernel();
}

void CacheBwBench::run_kernel() const {
    const size_t* psz = &n_;
    kernel_read(data_, psz);
}

void CacheBwBench::run_benchmark(int reps) {
    reps *= reps_mult_;
    struct timeval t_start, t_end;

    const double mono_ts = capture_monotonic();

    gettimeofday(&t_start, NULL);
    for (int r = 0; r < reps; ++r) {
        run_kernel();
    }
    gettimeofday(&t_end, NULL);

    const double t = sec(t_start, t_end);

    /* pure read: n_ * sizeof(double) bytes transferred per rep */
    const double bytes_per_rep = static_cast<double>(n_) * sizeof(double);
    const double bw_gbs = (bytes_per_rep * reps / 1.0e9) / t;

    PerfSample s;
    s.bench_name       = bench_name_;
    s.reps             = reps;
    s.monotonic_time_s = mono_ts;
    s.elapsed_s        = t;
    s.metrics.push_back({"array_size_MB", size_bytes_ / (1024.0*1024.0)});
    s.metrics.push_back({"n_elements", static_cast<double>(n_)});
    s.metrics.push_back({"bandwidth_GBs", bw_gbs});
    perf_data_.push_back(s);
}

void CacheBwBench::destroy_benchmark() {
    free(data_); data_ = nullptr;
    n_ = 0;
}
