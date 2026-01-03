/******************************************************************************
** Copyright (c) 2025, Alexander Heinecke                                    **
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

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
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

inline double sec(struct timeval start, struct timeval end) {
  return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))) / 1.0e6;
}

void run_benchmark( double* i_data, size_t i_arraySize, size_t i_copies, size_t i_iters ) {
  #pragma omp parallel shared(i_data, i_arraySize, i_copies, i_iters)
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

#ifdef __AVX512F__
#ifndef AVX512_BCAST128
#ifndef AVX512_BCAST256
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
#endif
#endif
#ifdef AVX512_BCAST128
      __asm__ __volatile__("movq %0, %%r8\n\t"
                           "movq %1, %%r10\n\t"
                           "movq (%%r10), %%r9\n\t"
                           "1:\n\t"
                           "subq $64, %%r9\n\t"
#if 0
                           "prefetcht0 2048(%%r8)\n\t"
#endif
                           "vbroadcasti32x4    0(%%r8),   %%zmm0\n\t"
                           "vbroadcasti32x4   16(%%r8),   %%zmm1\n\t"
                           "vbroadcasti32x4   32(%%r8),   %%zmm2\n\t"
                           "vbroadcasti32x4   48(%%r8),   %%zmm3\n\t"
#if 0
                           "prefetcht0 2112(%%r8)\n\t"
#endif
                           "vbroadcasti32x4   64(%%r8),   %%zmm4\n\t"
                           "vbroadcasti32x4   80(%%r8),   %%zmm5\n\t"
                           "vbroadcasti32x4   96(%%r8),   %%zmm6\n\t"
                           "vbroadcasti32x4  112(%%r8),   %%zmm7\n\t"
#if 0
                           "prefetcht0 2176(%%r8)\n\t"
#endif
                           "vbroadcasti32x4  128(%%r8),   %%zmm8\n\t"
                           "vbroadcasti32x4  144(%%r8),   %%zmm9\n\t"
                           "vbroadcasti32x4  160(%%r8),  %%zmm10\n\t"
                           "vbroadcasti32x4  176(%%r8),  %%zmm11\n\t"
#if 0
                           "prefetcht0 2240(%%r8)\n\t"
#endif
                           "vbroadcasti32x4  192(%%r8),  %%zmm12\n\t"
                           "vbroadcasti32x4  208(%%r8),  %%zmm13\n\t"
                           "vbroadcasti32x4  224(%%r8),  %%zmm14\n\t"
                           "vbroadcasti32x4  240(%%r8),  %%zmm15\n\t"
#if 0
                           "prefetcht0 2304(%%r8)\n\t"
#endif
                           "vbroadcasti32x4  256(%%r8),  %%zmm16\n\t"
                           "vbroadcasti32x4  272(%%r8),  %%zmm17\n\t"
                           "vbroadcasti32x4  288(%%r8),  %%zmm18\n\t"
                           "vbroadcasti32x4  304(%%r8),  %%zmm19\n\t"
#if 0
                           "prefetcht0 2368(%%r8)\n\t"
#endif
                           "vbroadcasti32x4  320(%%r8),  %%zmm20\n\t"
                           "vbroadcasti32x4  336(%%r8),  %%zmm21\n\t"
                           "vbroadcasti32x4  352(%%r8),  %%zmm22\n\t"
                           "vbroadcasti32x4  368(%%r8),  %%zmm23\n\t"
#if 0
                           "prefetcht0 2432(%%r8)\n\t"
#endif
                           "vbroadcasti32x4  384(%%r8),  %%zmm24\n\t"
                           "vbroadcasti32x4  400(%%r8),  %%zmm25\n\t"
                           "vbroadcasti32x4  416(%%r8),  %%zmm26\n\t"
                           "vbroadcasti32x4  432(%%r8),  %%zmm27\n\t"
#if 0
                           "prefetcht0 2496(%%r8)\n\t"
#endif
                           "vbroadcasti32x4  448(%%r8),  %%zmm28\n\t"
                           "vbroadcasti32x4  464(%%r8),  %%zmm29\n\t"
                           "vbroadcasti32x4  480(%%r8),  %%zmm30\n\t"
                           "vbroadcasti32x4  496(%%r8),  %%zmm31\n\t"
                           "addq $512, %%r8\n\t"
                           "cmpq $0, %%r9\n\t"
                           "jg 1b\n\t"
                      : : "m"(l_locAddr), "m"(l_parraySize)  : "r8","r9","r10","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","xmm16","xmm17","xmm18","xmm19","xmm20","xmm21","xmm22","xmm23","xmm24","xmm25","xmm26","xmm27","xmm28","xmm29","xmm30","xmm31");
#endif
#ifdef AVX512_BCAST256
      __asm__ __volatile__("movq %0, %%r8\n\t"
                           "movq %1, %%r10\n\t"
                           "movq (%%r10), %%r9\n\t"
                           "1:\n\t"
                           "subq $128, %%r9\n\t"
                           "vbroadcasti32x8    0(%%r8),   %%zmm0\n\t"
                           "vbroadcasti32x8   32(%%r8),   %%zmm1\n\t"
                           "vbroadcasti32x8   64(%%r8),   %%zmm2\n\t"
                           "vbroadcasti32x8   96(%%r8),   %%zmm3\n\t"
                           "vbroadcasti32x8  128(%%r8),   %%zmm4\n\t"
                           "vbroadcasti32x8  160(%%r8),   %%zmm5\n\t"
                           "vbroadcasti32x8  192(%%r8),   %%zmm6\n\t"
                           "vbroadcasti32x8  224(%%r8),   %%zmm7\n\t"
                           "vbroadcasti32x8  256(%%r8),   %%zmm8\n\t"
                           "vbroadcasti32x8  288(%%r8),   %%zmm9\n\t"
                           "vbroadcasti32x8  320(%%r8),  %%zmm10\n\t"
                           "vbroadcasti32x8  352(%%r8),  %%zmm11\n\t"
                           "vbroadcasti32x8  384(%%r8),  %%zmm12\n\t"
                           "vbroadcasti32x8  416(%%r8),  %%zmm13\n\t"
                           "vbroadcasti32x8  448(%%r8),  %%zmm14\n\t"
                           "vbroadcasti32x8  480(%%r8),  %%zmm15\n\t"
                           "vbroadcasti32x8  512(%%r8),  %%zmm16\n\t"
                           "vbroadcasti32x8  544(%%r8),  %%zmm17\n\t"
                           "vbroadcasti32x8  576(%%r8),  %%zmm18\n\t"
                           "vbroadcasti32x8  608(%%r8),  %%zmm19\n\t"
                           "vbroadcasti32x8  640(%%r8),  %%zmm20\n\t"
                           "vbroadcasti32x8  672(%%r8),  %%zmm21\n\t"
                           "vbroadcasti32x8  704(%%r8),  %%zmm22\n\t"
                           "vbroadcasti32x8  736(%%r8),  %%zmm23\n\t"
                           "vbroadcasti32x8  768(%%r8),  %%zmm24\n\t"
                           "vbroadcasti32x8  800(%%r8),  %%zmm25\n\t"
                           "vbroadcasti32x8  832(%%r8),  %%zmm26\n\t"
                           "vbroadcasti32x8  864(%%r8),  %%zmm27\n\t"
                           "vbroadcasti32x8  896(%%r8),  %%zmm28\n\t"
                           "vbroadcasti32x8  928(%%r8),  %%zmm29\n\t"
                           "vbroadcasti32x8  960(%%r8),  %%zmm30\n\t"
                           "vbroadcasti32x8  992(%%r8),  %%zmm31\n\t"
                           "addq $1024, %%r8\n\t"
                           "cmpq $0, %%r9\n\t"
                           "jg 1b\n\t"
                      : : "m"(l_locAddr), "m"(l_parraySize)  : "r8","r9","r10","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","xmm16","xmm17","xmm18","xmm19","xmm20","xmm21","xmm22","xmm23","xmm24","xmm25","xmm26","xmm27","xmm28","xmm29","xmm30","xmm31");
#endif
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
#elif __SSE__
      __asm__ __volatile__("movl %0, %%eax\n\t"
                           "movl %1, %%ebx\n\t"
                           "movl (%%ebx), %%ebx\n\t"
                           "1:\n\t"
                           "subl $16, %%ebx\n\t"
                           "movaps    0(%%eax),   %%xmm0\n\t"
                           "movaps   16(%%eax),   %%xmm1\n\t"
                           "movaps   32(%%eax),   %%xmm2\n\t"
                           "movaps   48(%%eax),   %%xmm3\n\t"
                           "movaps   64(%%eax),   %%xmm4\n\t"
                           "movaps   80(%%eax),   %%xmm5\n\t"
                           "movaps   96(%%eax),   %%xmm6\n\t"
                           "movaps  112(%%eax),   %%xmm7\n\t"
                           "addl $128, %%eax\n\t"
                           "cmpl $0, %%ebx\n\t"
                           "jg 1b\n\t"
                      : : "m"(l_locAddr), "m"(l_arraySize)  : "eax","ebx","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7");
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

  l_arraySize_0 = atoi(argv[1]);
  l_arrayFactor = atoi(argv[2]);
  l_arraySteps = atoi(argv[3]);
  l_copies = atoi(argv[4]);
  l_iters_0 = atoi(argv[5]);

  if (l_arraySize_0 % 256 != 0) {
    printf("ERROR mod 256\n");
    exit(-1);
  }

  printf("Using %zu private Read Buffers\n", l_copies);
  l_arraySize_0 *= l_copies;
  printf("KiB-per-core-read,GiB/s,Time\n");

  double* l_data;
  size_t l_n = 0;
  double l_avgTime;
  struct timeval l_startTime, l_endTime;
  size_t l_arraySize = (i == 0) ? l_arraySize_0 : l_arraySize_0 * i * l_arrayFactor;
  size_t l_iters = (i == 0) ? l_iters_0 : l_iters_0 / (i * l_arrayFactor);
  double l_size = (double)((size_t)l_arraySize*sizeof(double));
  // init data
  posix_memalign((void**)&l_data, 2097152, ((size_t)l_arraySize)*sizeof(double));;

  for ( l_n = 0; l_n < l_arraySize; l_n++ ) {
    l_data[l_n] = (double)l_n;
  }

  // pre-heat caches
  run_benchmark( l_data, l_arraySize, l_copies, 5 );

  // run benchmark
  gettimeofday(&l_startTime, NULL);
  run_benchmark( l_data, l_arraySize, l_copies, l_iters );
  gettimeofday(&l_endTime, NULL);
  l_avgTime = sec(l_startTime, l_endTime);
  // postprocess timing
  l_avgTime /= (double)l_iters;

  // output
  printf("%f,%f,%f\n", (l_size/l_copies)/1024.0, (((l_size/l_copies)*l_numThreads)/(1024.0*1024.0*1024.0))/l_avgTime, l_avgTime);
  free(l_data);

  return 0;
}
