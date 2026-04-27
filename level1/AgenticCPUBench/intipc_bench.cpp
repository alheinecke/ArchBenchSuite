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

#include "intipc_bench.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>

/* =========================================================================
 * Platform inline-assembly kernel
 *
 * Conceptual C reference:
 *   for (iter = 0; iter < num_iter; ++iter)
 *     for (i = 0; i < N; ++i)            // N = 4096
 *       for (j = 0; j < M; ++j)          // M = 8, fully unrolled
 *         accum[j] += data[j][i];
 *
 * Layout: data[j][i] = *(row_ptr[j] + i)   where row_ptr[j] points to
 * the base of row j and element stride is sizeof(int64_t) = 8 bytes.
 *
 * The i-loop is unrolled 4× so that the processor's decode/issue window
 * sees 4 × 8 = 32 independent load+add pairs per cycle group.
 * N=4096 is divisible by 4, so no tail handling is required.
 *
 * Register allocation (x86_64):
 *   Accumulators  : acc0..acc7  – 8 named "+r" operands (GCC allocated)
 *   Base pointer  : data_       – 1 named "r" input  (= row-0 base)
 *   num_iter      :             – 1 named "r" input
 *   iter          : r10         – internally clobbered
 *   i (col index) : r11         – internally clobbered
 *   Total GPRs    : 10 named + 2 clobbered = 12  (fits in 16)
 *
 *   Row j is accessed via a compile-time 32-bit displacement:
 *     j * N * sizeof(int64_t) = j * 32768 bytes
 *   so "addq j*32768(%[base], %%r11, 8), %[aj]" needs no extra register.
 *
 * Register allocation (AArch64):
 *   Accumulators  : acc0..acc7  – x0-x7  (caller-saved, allocated by GCC)
 *   Row pointers  : rp0..rp7   – x8-x15 (allocated by GCC)
 *   i (column idx): x16
 *   iter counter  : x17
 * =========================================================================*/

static constexpr int64_t M_BENCH = 8;
static constexpr int64_t N_BENCH = 4096;
/* i-loop unroll factor – N must be divisible by this value */
static constexpr int64_t UNROLL  = 4;

void IntIpcBench::run_kernel(int64_t num_iter) const {
    /* Pre-compute row base pointers (byte arithmetic in C, avoid large
     * displacements in the asm).  Each row has N_BENCH int64_t elements. */
    const int64_t* rp0 = data_ + 0 * N_BENCH;
    const int64_t* rp1 = data_ + 1 * N_BENCH;
    const int64_t* rp2 = data_ + 2 * N_BENCH;
    const int64_t* rp3 = data_ + 3 * N_BENCH;
    const int64_t* rp4 = data_ + 4 * N_BENCH;
    const int64_t* rp5 = data_ + 5 * N_BENCH;
    const int64_t* rp6 = data_ + 6 * N_BENCH;
    const int64_t* rp7 = data_ + 7 * N_BENCH;

    /* Accumulators, initialised from accum_ so the compiler cannot treat
     * them as dead and eliminate the asm block.                           */
    int64_t a0 = accum_[0], a1 = accum_[1], a2 = accum_[2], a3 = accum_[3];
    int64_t a4 = accum_[4], a5 = accum_[5], a6 = accum_[6], a7 = accum_[7];

/* -------------------------------------------------------------------------
 * x86_64 kernel
 * -------------------------------------------------------------------------
 * Single base pointer: row j lives at byte offset j*N*8 = j*32768 from
 * data_.  All offsets fit in a 32-bit signed displacement, so the
 * addressing mode  disp(%[base], %%r11, 8)  introduces no extra register
 * per row.  N=4096 is encoded as the immediate $4096.
 *
 * Outer loop  : iter  (r10)
 * Inner loop  : i     (r11), element index, step = UNROLL = 4
 * -------------------------------------------------------------------------*/
#if defined(__x86_64__)
    __asm__ volatile (
        /* ---- outer (iter) loop ---- */
        "xorq %%r10, %%r10\n\t"                 /* r10 = iter = 0          */
        ".p2align 4\n\t"
        "1:\n\t"
        "  cmpq %[num_iter], %%r10\n\t"
        "  jge 3f\n\t"

        /* ---- inner (i) loop, unrolled 4× ---- */
        "  xorq %%r11, %%r11\n\t"               /* r11 = i = 0             */
        ".p2align 4\n\t"
        "2:\n\t"
        "  cmpq $4096, %%r11\n\t"               /* N hardcoded             */
        "  jge 4f\n\t"

        /* -- slot 0: col i -- */
        "  addq      0(%[base], %%r11, 8), %[a0]\n\t"
        "  addq  32768(%[base], %%r11, 8), %[a1]\n\t"
        "  addq  65536(%[base], %%r11, 8), %[a2]\n\t"
        "  addq  98304(%[base], %%r11, 8), %[a3]\n\t"
        "  addq 131072(%[base], %%r11, 8), %[a4]\n\t"
        "  addq 163840(%[base], %%r11, 8), %[a5]\n\t"
        "  addq 196608(%[base], %%r11, 8), %[a6]\n\t"
        "  addq 229376(%[base], %%r11, 8), %[a7]\n\t"

        /* -- slot 1: col i+1  (each displacement += 8) -- */
        "  addq      8(%[base], %%r11, 8), %[a0]\n\t"
        "  addq  32776(%[base], %%r11, 8), %[a1]\n\t"
        "  addq  65544(%[base], %%r11, 8), %[a2]\n\t"
        "  addq  98312(%[base], %%r11, 8), %[a3]\n\t"
        "  addq 131080(%[base], %%r11, 8), %[a4]\n\t"
        "  addq 163848(%[base], %%r11, 8), %[a5]\n\t"
        "  addq 196616(%[base], %%r11, 8), %[a6]\n\t"
        "  addq 229384(%[base], %%r11, 8), %[a7]\n\t"

        /* -- slot 2: col i+2  (each displacement += 16) -- */
        "  addq     16(%[base], %%r11, 8), %[a0]\n\t"
        "  addq  32784(%[base], %%r11, 8), %[a1]\n\t"
        "  addq  65552(%[base], %%r11, 8), %[a2]\n\t"
        "  addq  98320(%[base], %%r11, 8), %[a3]\n\t"
        "  addq 131088(%[base], %%r11, 8), %[a4]\n\t"
        "  addq 163856(%[base], %%r11, 8), %[a5]\n\t"
        "  addq 196624(%[base], %%r11, 8), %[a6]\n\t"
        "  addq 229392(%[base], %%r11, 8), %[a7]\n\t"

        /* -- slot 3: col i+3  (each displacement += 24) -- */
        "  addq     24(%[base], %%r11, 8), %[a0]\n\t"
        "  addq  32792(%[base], %%r11, 8), %[a1]\n\t"
        "  addq  65560(%[base], %%r11, 8), %[a2]\n\t"
        "  addq  98328(%[base], %%r11, 8), %[a3]\n\t"
        "  addq 131096(%[base], %%r11, 8), %[a4]\n\t"
        "  addq 163864(%[base], %%r11, 8), %[a5]\n\t"
        "  addq 196632(%[base], %%r11, 8), %[a6]\n\t"
        "  addq 229400(%[base], %%r11, 8), %[a7]\n\t"

        "  addq $4, %%r11\n\t"                  /* i += UNROLL             */
        "  jmp 2b\n\t"
        "4:\n\t"

        "  incq %%r10\n\t"                       /* ++iter                  */
        "  jmp 1b\n\t"
        "3:\n\t"

        : /* outputs */
          [a0] "+r" (a0), [a1] "+r" (a1), [a2] "+r" (a2), [a3] "+r" (a3),
          [a4] "+r" (a4), [a5] "+r" (a5), [a6] "+r" (a6), [a7] "+r" (a7)
        : /* inputs */
          [base] "r" (data_), [num_iter] "r" (num_iter)
        : /* clobbers */
          "r10", "r11", "cc", "memory"
    );

/* -------------------------------------------------------------------------
 * AArch64 kernel
 * -------------------------------------------------------------------------
 * Outer loop  : iter  (x16)
 * Inner loop  : i     (x17), counts in bytes (stride 8*UNROLL = 32)
 * Temporaries : x18..x21 hold loads for the 4 unroll slots per row
 *
 * ldr x_tmp, [rp, i]   then   add acc, acc, x_tmp
 * AArch64 has no memory-operand add, so explicit load registers are used.
 * x18-x30 are caller/callee saved; GCC reserves x18 on some platforms as
 * the platform register – we use x19-x26 for 8 temporaries to be safe.
 *
 * i counts in bytes so that "ldr x_t, [xRp, xI]" works directly.
 * Step per unrolled iteration = UNROLL * sizeof(int64_t) = 32 bytes.
 * Loop bound = N_BENCH * sizeof(int64_t) = 32768 bytes.
 * -------------------------------------------------------------------------*/
#elif defined(__aarch64__)
    __asm__ volatile (
        /* ---- outer (iter) loop ---- */
        "mov x16, #0\n\t"                        /* x16 = iter = 0         */
        "1:\n\t"
        "cmp x16, %[num_iter]\n\t"
        "bge 3f\n\t"

        /* ---- inner (i) loop, unrolled 4×, i in bytes ---- */
        "mov x17, #0\n\t"                        /* x17 = byte offset = 0  */
        "2:\n\t"
        "cmp x17, %[N_bytes]\n\t"
        "bge 4f\n\t"

        /* -- unroll slot 0: byte offset i+0 -- */
        "ldr x19, [%[rp0], x17]\n\t"
        "ldr x20, [%[rp1], x17]\n\t"
        "ldr x21, [%[rp2], x17]\n\t"
        "ldr x22, [%[rp3], x17]\n\t"
        "ldr x23, [%[rp4], x17]\n\t"
        "ldr x24, [%[rp5], x17]\n\t"
        "ldr x25, [%[rp6], x17]\n\t"
        "ldr x26, [%[rp7], x17]\n\t"
        "add %[a0], %[a0], x19\n\t"
        "add %[a1], %[a1], x20\n\t"
        "add %[a2], %[a2], x21\n\t"
        "add %[a3], %[a3], x22\n\t"
        "add %[a4], %[a4], x23\n\t"
        "add %[a5], %[a5], x24\n\t"
        "add %[a6], %[a6], x25\n\t"
        "add %[a7], %[a7], x26\n\t"

        /* -- unroll slot 1: byte offset i+8 -- */
        "add x9, x17, #8\n\t"
        "ldr x19, [%[rp0], x9]\n\t"
        "ldr x20, [%[rp1], x9]\n\t"
        "ldr x21, [%[rp2], x9]\n\t"
        "ldr x22, [%[rp3], x9]\n\t"
        "ldr x23, [%[rp4], x9]\n\t"
        "ldr x24, [%[rp5], x9]\n\t"
        "ldr x25, [%[rp6], x9]\n\t"
        "ldr x26, [%[rp7], x9]\n\t"
        "add %[a0], %[a0], x19\n\t"
        "add %[a1], %[a1], x20\n\t"
        "add %[a2], %[a2], x21\n\t"
        "add %[a3], %[a3], x22\n\t"
        "add %[a4], %[a4], x23\n\t"
        "add %[a5], %[a5], x24\n\t"
        "add %[a6], %[a6], x25\n\t"
        "add %[a7], %[a7], x26\n\t"

        /* -- unroll slot 2: byte offset i+16 -- */
        "add x9, x17, #16\n\t"
        "ldr x19, [%[rp0], x9]\n\t"
        "ldr x20, [%[rp1], x9]\n\t"
        "ldr x21, [%[rp2], x9]\n\t"
        "ldr x22, [%[rp3], x9]\n\t"
        "ldr x23, [%[rp4], x9]\n\t"
        "ldr x24, [%[rp5], x9]\n\t"
        "ldr x25, [%[rp6], x9]\n\t"
        "ldr x26, [%[rp7], x9]\n\t"
        "add %[a0], %[a0], x19\n\t"
        "add %[a1], %[a1], x20\n\t"
        "add %[a2], %[a2], x21\n\t"
        "add %[a3], %[a3], x22\n\t"
        "add %[a4], %[a4], x23\n\t"
        "add %[a5], %[a5], x24\n\t"
        "add %[a6], %[a6], x25\n\t"
        "add %[a7], %[a7], x26\n\t"

        /* -- unroll slot 3: byte offset i+24 -- */
        "add x9, x17, #24\n\t"
        "ldr x19, [%[rp0], x9]\n\t"
        "ldr x20, [%[rp1], x9]\n\t"
        "ldr x21, [%[rp2], x9]\n\t"
        "ldr x22, [%[rp3], x9]\n\t"
        "ldr x23, [%[rp4], x9]\n\t"
        "ldr x24, [%[rp5], x9]\n\t"
        "ldr x25, [%[rp6], x9]\n\t"
        "ldr x26, [%[rp7], x9]\n\t"
        "add %[a0], %[a0], x19\n\t"
        "add %[a1], %[a1], x20\n\t"
        "add %[a2], %[a2], x21\n\t"
        "add %[a3], %[a3], x22\n\t"
        "add %[a4], %[a4], x23\n\t"
        "add %[a5], %[a5], x24\n\t"
        "add %[a6], %[a6], x25\n\t"
        "add %[a7], %[a7], x26\n\t"

        "add x17, x17, #32\n\t"                  /* i += 4 * 8 bytes       */
        "b 2b\n\t"
        "4:\n\t"

        "add x16, x16, #1\n\t"                   /* ++iter                 */
        "b 1b\n\t"
        "3:\n\t"

        : /* outputs */
          [a0] "+r" (a0), [a1] "+r" (a1), [a2] "+r" (a2), [a3] "+r" (a3),
          [a4] "+r" (a4), [a5] "+r" (a5), [a6] "+r" (a6), [a7] "+r" (a7)
        : /* inputs */
          [rp0] "r" (rp0), [rp1] "r" (rp1), [rp2] "r" (rp2), [rp3] "r" (rp3),
          [rp4] "r" (rp4), [rp5] "r" (rp5), [rp6] "r" (rp6), [rp7] "r" (rp7),
          [num_iter] "r" (num_iter),
          [N_bytes] "r" (N_BENCH * (int64_t)sizeof(int64_t))
        : /* clobbers */
          "x16", "x17", "x9", "x19", "x20", "x21", "x22",
          "x23", "x24", "x25", "x26", "memory"
    );

#else
    /* -----------------------------------------------------------------------
     * Generic C fallback for unsupported architectures.
     * ----------------------------------------------------------------------- */
    for (int64_t iter = 0; iter < num_iter; ++iter) {
        for (int64_t i = 0; i < N_BENCH; ++i) {
            a0 += rp0[i]; a1 += rp1[i]; a2 += rp2[i]; a3 += rp3[i];
            a4 += rp4[i]; a5 += rp5[i]; a6 += rp6[i]; a7 += rp7[i];
        }
    }
#endif

    /* Write accumulators back so the compiler cannot treat the asm as dead */
    const_cast<IntIpcBench*>(this)->accum_[0] = a0;
    const_cast<IntIpcBench*>(this)->accum_[1] = a1;
    const_cast<IntIpcBench*>(this)->accum_[2] = a2;
    const_cast<IntIpcBench*>(this)->accum_[3] = a3;
    const_cast<IntIpcBench*>(this)->accum_[4] = a4;
    const_cast<IntIpcBench*>(this)->accum_[5] = a5;
    const_cast<IntIpcBench*>(this)->accum_[6] = a6;
    const_cast<IntIpcBench*>(this)->accum_[7] = a7;
}

/* =========================================================================
 * IntIpcBench public interface
 * =========================================================================*/

IntIpcBench::IntIpcBench() = default;

IntIpcBench::~IntIpcBench() {
    destroy_benchmark();
}

void IntIpcBench::setup_benchmark(size_t size_bytes) {
    size_bytes_ = size_bytes;
    num_iter_   = static_cast<int64_t>(size_bytes / (1024ULL * 1024ULL)) * NUM_ITER_SCALE;

    data_ = static_cast<int64_t*>(
        aligned_alloc(64, static_cast<size_t>(M_BENCH * N_BENCH) * sizeof(int64_t)));

    if (!data_) {
        std::cerr << "IntIpcBench::setup_benchmark: memory allocation failed."
                  << std::endl;
        return;
    }

    /* Fill data with small non-zero values to avoid trivial optimisation */
    srand(42);
    for (int64_t k = 0; k < M_BENCH * N_BENCH; ++k)
        data_[k] = static_cast<int64_t>(rand() % 256) + 1;

    /* Zero accumulators */
    for (int64_t j = 0; j < M_BENCH; ++j)
        accum_[j] = 0;

    /* Warm-up: two passes to prime caches */
    run_kernel(2);
}

void IntIpcBench::run_benchmark(int reps) {
    reps *= reps_mult_;
    if (!data_) {
        std::cerr << "IntIpcBench::run_benchmark: benchmark not initialised."
                  << std::endl;
        return;
    }

    struct timeval t_start, t_end;

    const double mono_ts = capture_monotonic();

    gettimeofday(&t_start, NULL);
    for (int r = 0; r < reps; ++r) {
        run_kernel(num_iter_);
    }
    gettimeofday(&t_end, NULL);

    const double t = sec(t_start, t_end);

    /* ops = M * N adds per outer iteration, times reps */
    const double total_ops = static_cast<double>(num_iter_) * M_BENCH * N_BENCH * reps;
    const double gops      = (total_ops / 1.0e9) / t;

    PerfSample s;
    s.bench_name       = bench_name_;
    s.reps             = reps;
    s.monotonic_time_s = mono_ts;
    s.elapsed_s        = t;
    s.metrics.push_back({"M", static_cast<double>(M_BENCH)});
    s.metrics.push_back({"N", static_cast<double>(N_BENCH)});
    s.metrics.push_back({"num_iter", static_cast<double>(num_iter_)});
    s.metrics.push_back({"GOPS", gops});
    perf_data_.push_back(s);
}

void IntIpcBench::destroy_benchmark() {
    free(data_);
    data_     = nullptr;
    num_iter_ = 0;
    size_bytes_ = 0;
    for (int64_t j = 0; j < M_BENCH; ++j)
        accum_[j] = 0;
}
