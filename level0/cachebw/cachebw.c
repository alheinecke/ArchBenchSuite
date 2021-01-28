/******************************************************************************
** Copyright (c) 2013-2020, Alexander Heinecke                               **
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

#if 0
#define USE_PERF_COUNTERS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_PERF_COUNTERS
#include "./../common/counters.h"
#endif

#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES	10000000
#endif
#endif
#ifndef NTIMES
#   define NTIMES	10000000
#endif

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#ifdef __VSX__
void run_gpr_kernel(size_t* i_data, size_t i_chunkSize) {
        __asm__ __volatile__("li 12, 0x0\n\t"
                             "1:\n\t"
                             "ld 29,  0(%0)\n\t"
                             "ld 28,  8(%0)\n\t"
                             "ld 27, 16(%0)\n\t"
                             "ld 26, 24(%0)\n\t"
                             "ld 25, 32(%0)\n\t"
                             "ld 24, 40(%0)\n\t"
                             "ld 23, 48(%0)\n\t"
                             "ld 22, 56(%0)\n\t"
                             "ld 21, 64(%0)\n\t"
                             "ld 20, 72(%0)\n\t"
                             "ld 19, 80(%0)\n\t"
                             "ld 18, 88(%0)\n\t"
                             "ld 17, 96(%0)\n\t"
                             "ld 16, 104(%0)\n\t"
                             "ld 15, 112(%0)\n\t"
                             "ld 14, 120(%0)\n\t"
                             "add %0, %0, 12\n\t"
                             "subic. %1, %1, 0x10\n\t"
                             "bne 1b\n\t"
                       : : "r"(i_data), "r"(i_chunkSize)  : "r12","r14","r15","r16","r17","r18","r19","r20","r21","r22","r23","r24","r25","r26","r27","r28","r29");
}

void run_vsx_kernel(double* i_data, size_t i_chunkSize) {
        __asm__ __volatile__("li 27, 0x10\n\t"
                             "li 26, 0x20\n\t"
                             "li 25, 0x30\n\t"
                             "li 24, 0x40\n\t"
                             "li 23, 0x50\n\t"
                             "li 22, 0x60\n\t"
                             "li 21, 0x70\n\t"
                             "li 20, 0x80\n\t"
                             "1:\n\t"
                             "lxvd2x 63, 0, %0\n\t"
                             "lxvd2x 62, 27, %0\n\t"
                             "lxvd2x 61, 26, %0\n\t"
                             "lxvd2x 60, 25, %0\n\t"
                             "lxvd2x 59, 24, %0\n\t"
                             "lxvd2x 58, 23, %0\n\t"
                             "lxvd2x 57, 22, %0\n\t"
                             "lxvd2x 56, 21, %0\n\t"
                             "add %0, %0, 20\n\t"
                             "lxvd2x 55, 0, %0\n\t"
                             "lxvd2x 54, 27, %0\n\t"
                             "lxvd2x 53, 26, %0\n\t"
                             "lxvd2x 52, 25, %0\n\t"
                             "lxvd2x 51, 24, %0\n\t"
                             "lxvd2x 50, 23, %0\n\t"
                             "lxvd2x 49, 22, %0\n\t"
                             "lxvd2x 48, 21, %0\n\t"
                             "add %0, %0, 20\n\t"
                             "subic. %1, %1, 0x20\n\t"
                             "bne 1b\n\t"
                       : : "r"(i_data), "r"(i_chunkSize)  : "r20", "r21","r22","r23","r24","r25","r26","r27","vs63","vs62","vs61","vs60","vs59","vs58","vs57","vs56","vs55","vs54","vs53","vs52","vs51","vs50","vs49","vs48");
}
#endif

inline double sec(struct timeval start, struct timeval end) {
  return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))) / 1.0e6;
}


inline void run_benchmark( double* i_data, size_t i_arraySize, size_t i_copies, size_t i_iters ) {
  // we do manual reduction here as we don't rely on a smart OpenMP implementation
  #pragma omp parallel 
  {
#ifdef _OPENMP
    size_t l_tid = omp_get_thread_num();
    size_t l_numThreads = omp_get_num_threads();
#else
    size_t l_tid = 0;
    size_t l_numThreads = 1;
#endif
    size_t  l_arraySize = i_arraySize/i_copies;
    size_t  threads_per_copy = l_numThreads / i_copies;
    double* l_locAddr = i_data + (l_arraySize*(l_tid/threads_per_copy));
    size_t* l_parraySize = &(l_arraySize);
    size_t  l_i = 0;

    for( l_i = 0; l_i < i_iters; l_i++ ) {
#ifdef __VSX__
      //if (l_tid % 2 == 0) {
      run_vsx_kernel(l_locAddr, l_arraySize);
      //} else {
      //  run_gpr_kernel((size_t*)l_locAddr, l_chunkSize);
      //}
#endif
#ifdef __ARM_NEON
        __asm__ __volatile__("mov x0, %0\n\t"
                             "mov x1, %1\n\t""
                             "1:\n\t"
                             "ld1  {v0.2d}, [x0],16\n\t"
                             "ld1  {v1.2d}, [x0],16\n\t"
                             "ld1  {v2.2d}, [x0],16\n\t"
                             "ld1  {v3.2d}, [x0],16\n\t"
                             "ld1  {v4.2d}, [x0],16\n\t"
                             "ld1  {v5.2d}, [x0],16\n\t"
                             "ld1  {v6.2d}, [x0],16\n\t"
                             "ld1  {v7.2d}, [x0],16\n\t"
                             "ld1  {v8.2d}, [x0],16\n\t"
                             "ld1  {v9.2d}, [x0],16\n\t"
                             "ld1 {v10.2d}, [x0],16\n\t"
                             "ld1 {v11.2d}, [x0],16\n\t"
                             "ld1 {v12.2d}, [x0],16\n\t"
                             "ld1 {v13.2d}, [x0],16\n\t"
                             "ld1 {v14.2d}, [x0],16\n\t"
                             "ld1 {v15.2d}, [x0],16\n\t"
                             "ld1 {v16.2d}, [x0],16\n\t"
                             "ld1 {v17.2d}, [x0],16\n\t"
                             "ld1 {v18.2d}, [x0],16\n\t"
                             "ld1 {v19.2d}, [x0],16\n\t"
                             "ld1 {v20.2d}, [x0],16\n\t"
                             "ld1 {v21.2d}, [x0],16\n\t"
                             "ld1 {v22.2d}, [x0],16\n\t"
                             "ld1 {v23.2d}, [x0],16\n\t"
                             "ld1 {v24.2d}, [x0],16\n\t"
                             "ld1 {v25.2d}, [x0],16\n\t"
                             "ld1 {v26.2d}, [x0],16\n\t"
                             "ld1 {v27.2d}, [x0],16\n\t"
                             "ld1 {v28.2d}, [x0],16\n\t"
                             "ld1 {v29.2d}, [x0],16\n\t"
                             "ld1 {v30.2d}, [x0],16\n\t"
                             "ld1 {v31.2d}, [x0],16\n\t"
                             "sub x1, x1, #64\n\t"
                             "cbnz x1, 1b\n\t"
                        : : "r" (l_locAddr), "r" (l_parraySize) : "x0","x1","v0","v1","v2","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31"); 
#endif
#ifdef __AVX512F__
      __asm__ __volatile__("movq %0, %%r8\n\t"
                           "movq %1, %%r10\n\t"
                           "movq (%%r10), %%r9\n\t"
                           "1:\n\t"
                           "subq $256, %%r9\n\t"
                           "vmovapd     0(%%r8),   %%zmm0\n\t"
                           "vmovapd    64(%%r8),   %%zmm1\n\t"
                           "vmovapd   128(%%r8),   %%zmm2\n\t"
                           "vmovapd   192(%%r8),   %%zmm3\n\t"
                           "vmovapd   256(%%r8),   %%zmm4\n\t"
                           "vmovapd   320(%%r8),   %%zmm5\n\t"
                           "vmovapd   384(%%r8),   %%zmm6\n\t"
                           "vmovapd   448(%%r8),   %%zmm7\n\t"
                           "vmovapd   512(%%r8),   %%zmm8\n\t"
                           "vmovapd   576(%%r8),   %%zmm9\n\t"
                           "vmovapd   640(%%r8),  %%zmm10\n\t"
                           "vmovapd   704(%%r8),  %%zmm11\n\t"
                           "vmovapd   768(%%r8),  %%zmm12\n\t"
                           "vmovapd   832(%%r8),  %%zmm13\n\t"
                           "vmovapd   896(%%r8),  %%zmm14\n\t"
                           "vmovapd   960(%%r8),  %%zmm15\n\t"
                           "vmovapd  1024(%%r8),  %%zmm16\n\t"
                           "vmovapd  1088(%%r8),  %%zmm17\n\t"
                           "vmovapd  1152(%%r8),  %%zmm18\n\t"
                           "vmovapd  1216(%%r8),  %%zmm19\n\t"
                           "vmovapd  1280(%%r8),  %%zmm20\n\t"
                           "vmovapd  1344(%%r8),  %%zmm21\n\t"
                           "vmovapd  1408(%%r8),  %%zmm22\n\t"
                           "vmovapd  1472(%%r8),  %%zmm23\n\t"
                           "vmovapd  1536(%%r8),  %%zmm24\n\t"
                           "vmovapd  1600(%%r8),  %%zmm25\n\t"
                           "vmovapd  1664(%%r8),  %%zmm26\n\t"
                           "vmovapd  1728(%%r8),  %%zmm27\n\t"
                           "vmovapd  1792(%%r8),  %%zmm28\n\t"
                           "vmovapd  1856(%%r8),  %%zmm29\n\t"
                           "vmovapd  1920(%%r8),  %%zmm30\n\t"
                           "vmovapd  1984(%%r8),  %%zmm31\n\t"
                           "addq $2048, %%r8\n\t"
                           "cmpq $0, %%r9\n\t"
                           "jg 1b\n\t"
                      : : "m"(l_locAddr), "m"(l_parraySize)  : "r8","r9","r10","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","xmm16","xmm17","xmm18","xmm19","xmm20","xmm21","xmm22","xmm23","xmm24","xmm25","xmm26","xmm27","xmm28","xmm29","xmm30","xmm31");
#elif __AVX__
      __asm__ __volatile__("movq %0, %%r8\n\t"
                           "movq %1, %%r10\n\t"
                           "movq (%%r10), %%r9\n\t"
                           "1:\n\t"
                           "subq $64, %%r9\n\t"
                           "vmovapd    0(%%r8),   %%ymm0\n\t"
                           "vmovapd   32(%%r8),   %%ymm1\n\t"
                           "vmovapd   64(%%r8),   %%ymm2\n\t"
                           "vmovapd   96(%%r8),   %%ymm3\n\t"
                           "vmovapd  128(%%r8),   %%ymm4\n\t"
                           "vmovapd  160(%%r8),   %%ymm5\n\t"
                           "vmovapd  192(%%r8),   %%ymm6\n\t"
                           "vmovapd  224(%%r8),   %%ymm7\n\t"
                           "vmovapd  256(%%r8),   %%ymm8\n\t"
                           "vmovapd  288(%%r8),   %%ymm9\n\t"
                           "vmovapd  320(%%r8),  %%ymm10\n\t"
                           "vmovapd  352(%%r8),  %%ymm11\n\t"
                           "vmovapd  384(%%r8),  %%ymm12\n\t"
                           "vmovapd  416(%%r8),  %%ymm13\n\t"
                           "vmovapd  448(%%r8),  %%ymm14\n\t"
                           "vmovapd  480(%%r8),  %%ymm15\n\t"
                           "addq $512, %%r8\n\t"
                           "cmpq $0, %%r9\n\t"
                           "jg 1b\n\t"
                      : : "m"(l_locAddr), "m"(l_parraySize)  : "r8","r9","r10","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
#elif __SSE3__
      __asm__ __volatile__("movq %0, %%r8\n\t"
                           "movq %1, %%r10\n\t"
                           "movq (%%r10), %%r9\n\t"
                           "1:\n\t"
                           "subq $32, %%r9\n\t"
                           "movapd    0(%%r8),   %%xmm0\n\t"
                           "movapd   16(%%r8),   %%xmm1\n\t"
                           "movapd   32(%%r8),   %%xmm2\n\t"
                           "movapd   48(%%r8),   %%xmm3\n\t"
                           "movapd   64(%%r8),   %%xmm4\n\t"
                           "movapd   80(%%r8),   %%xmm5\n\t"
                           "movapd   96(%%r8),   %%xmm6\n\t"
                           "movapd  112(%%r8),   %%xmm7\n\t"
                           "movapd  128(%%r8),   %%xmm8\n\t"
                           "movapd  144(%%r8),   %%xmm9\n\t"
                           "movapd  160(%%r8),  %%xmm10\n\t"
                           "movapd  176(%%r8),  %%xmm11\n\t"
                           "movapd  192(%%r8),  %%xmm12\n\t"
                           "movapd  208(%%r8),  %%xmm13\n\t"
                           "movapd  224(%%r8),  %%xmm14\n\t"
                           "movapd  240(%%r8),  %%xmm15\n\t"
                           "addq $256, %%r8\n\t"
                           "cmpq $0, %%r9\n\t"
                           "jg 1b\n\t"
                      : : "m"(l_locAddr), "m"(l_parraySize)  : "r8","r9","r10","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
#endif
      }
    } 
} 


int main(int argc, char* argv[]) {
  size_t l_numThreads = 1;
  size_t l_arraySize_0 = 256;
  size_t l_arrayFactor = 2;
  size_t l_arraySteps = 1;
  size_t l_iters_0 = 1;
  size_t l_copies = 1;
  size_t i = 0;
#ifdef USE_PERF_COUNTERS
  ctrs_skx_uc a, b, s;
  bw_gibs bw_cnt;

  setup_skx_uc_ctrs( CTRS_EXP_CHA_LLC_LOOKUP_VICTIMS );
  zero_skx_uc_ctrs( &a );
  zero_skx_uc_ctrs( &b );
  zero_skx_uc_ctrs( &s );
#endif


  if (argc != 6) {
    printf("#doubles increase-factor increase-steps copies #reps\n");
    return -1;
  }

  l_arraySize_0 = atoi(argv[1]);
  l_arrayFactor = atoi(argv[2]);
  l_arraySteps = atoi(argv[3]); 
  l_copies = atoi(argv[4]);
  l_iters_0 = atoi(argv[5]);


#ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp master
    l_numThreads = omp_get_num_threads();
  }
#else
  l_numThreads = 1;
#endif

  if (l_arraySize_0 % 256 != 0) {
    printf("ERROR % 256\n");
    exit(-1);
  }
  
  printf("Number of threads: %lld\n", l_numThreads);
  printf("Using %i private Read Buffers\n", l_copies);
  l_arraySize_0 *= l_copies;
  printf("KiB-per-core-read,GiB/s,Time\n");
 
  for ( i = 0 ; i < l_arraySteps; ++i ) {
    double* l_data;
    size_t l_n = 0;
    double l_avgTime;
    struct timeval l_startTime, l_endTime;
    size_t l_arraySize = (i == 0) ? l_arraySize_0 : l_arraySize_0 * i * l_arrayFactor;
    size_t l_iters = (i == 0) ? l_iters_0 : l_iters_0 / (i * l_arrayFactor);
    double l_size = (double)((size_t)l_arraySize*sizeof(double));
    // init data
    posix_memalign((void**)&l_data, 2097152, ((size_t)l_arraySize)*sizeof(double));;

    #pragma omp parallel for private(l_n)
    for ( l_n = 0; l_n < l_arraySize; l_n++ ) {
      l_data[l_n] = (double)l_n;
    }

    // pre-heat caches
    run_benchmark( l_data, l_arraySize, l_copies, 5 );
    
    // run benchmark
#ifdef USE_PERF_COUNTERS
    read_skx_uc_ctrs( &a );
#endif
    gettimeofday(&l_startTime, NULL);
    run_benchmark( l_data, l_arraySize, l_copies, l_iters );
    gettimeofday(&l_endTime, NULL);
    l_avgTime = sec(l_startTime, l_endTime);
#ifdef USE_PERF_COUNTERS
    read_skx_uc_ctrs( &b );
    difa_skx_uc_ctrs( &a, &b, &s );
    divi_skx_uc_ctrs( &s, l_iters );
#endif
    // postprocess timing 
    l_avgTime /= (double)l_iters;
    
    // output
    printf("%f,%f,%f\n", (l_size/l_copies)/1024.0, (((l_size/l_copies)*l_numThreads)/(1024.0*1024.0*1024.0))/l_avgTime, l_avgTime);
#ifdef USE_PERF_COUNTERS
    get_llc_bw_skx( &s, l_avgTime, &bw_cnt );
    printf("%f,%f,%f,%f,%f (counters)\n", (l_size/l_copies)/1024.0, bw_cnt.rd, bw_cnt.wr, bw_cnt.wr2, l_avgTime);
#endif
    free(l_data);
  }
  return 0; 
}
