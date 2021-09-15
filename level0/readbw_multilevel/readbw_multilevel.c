/******************************************************************************
** Copyright (c) 2013-2021, Alexander Heinecke                               **
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
#define USE_CORE_PERF_SNP
#endif
#if 0
#define USE_CORE_PERF_L2IN
#endif
#if 0
#define USE_CORE_PERF_IPC
#endif
#if 0
#define USE_UNCORE_PERF_DRAM_BW
#endif
#if 0
#define USE_UNCORE_PERF_LLC_VICTIMS
#endif
#if 0
#define USE_UNCORE_PERF_CHA_UTIL
#endif
#if 0
#define USE_UNCORE_PREF_AK_UTIL
#endif
#if 0
#define USE_UNCORE_PREF_IV_UTIL
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#define MY_MIN(A,B) (((A)<(B))?(A):(B))

#if defined(USE_CORE_PERF_SNP) || defined(USE_CORE_PERF_L2IN) || defined(USE_CORE_PERF_IPC) || defined(USE_UNCORE_PERF_DRAM_BW) || defined(USE_UNCORE_PERF_LLC_VICTIMS) || defined(USE_UNCORE_PERF_CHA_UTIL) || defined(USE_UNCORE_PREF_AK_UTIL) || defined(USE_UNCORE_PREF_IV_UTIL)
#  include "../common/perf_counter_markers.h"
#endif

inline double sec(struct timeval start, struct timeval end) {
  return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))) / 1.0e6;
}

void read_buffer( char* i_buffer, size_t i_length ) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%r8\n\t"
                       "movq %1, %%r9\n\t"
                       "1:\n\t"
                       "subq $2048, %%r9\n\t"
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
                       : : "m"(i_buffer), "r"(i_length) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","xmm16","xmm17","xmm18","xmm19","xmm20","xmm21","xmm22","xmm23","xmm24","xmm25","xmm26","xmm27","xmm28","xmm29","xmm30","xmm31");
#elif __AVX__
  __asm__ __volatile__("movq %0, %%r8\n\t"
                       "movq %1, %%r9\n\t"
                       "1:\n\t"
                       "subq $512, %%r9\n\t"
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
                       : : "m"(i_buffer), "r"(i_length) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
#elif __SSE2__
  __asm__ __volatile__("movq %0, %%r8\n\t"
                       "movq %1, %%r9\n\t"
                       "1:\n\t"
                       "subq $256, %%r9\n\t"
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
                       : : "m"(i_buffer), "r"(i_length) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
#else
#error need at least SSE2
#endif
} 

void clflush_buffer( char* i_buffer, size_t i_length ) {
  __asm__ __volatile__("movq %0, %%r8\n\t"
                       "movq %1, %%r9\n\t"
                       "1:\n\t"
                       "subq $2048, %%r9\n\t"
                       "clflushopt     0(%%r8)\n\t"
                       "clflushopt    64(%%r8)\n\t"
                       "clflushopt   128(%%r8)\n\t"
                       "clflushopt   192(%%r8)\n\t"
                       "clflushopt   256(%%r8)\n\t"
                       "clflushopt   320(%%r8)\n\t"
                       "clflushopt   384(%%r8)\n\t"
                       "clflushopt   448(%%r8)\n\t"
                       "clflushopt   512(%%r8)\n\t"
                       "clflushopt   576(%%r8)\n\t"
                       "clflushopt   640(%%r8)\n\t"
                       "clflushopt   704(%%r8)\n\t"
                       "clflushopt   768(%%r8)\n\t"
                       "clflushopt   832(%%r8)\n\t"
                       "clflushopt   896(%%r8)\n\t"
                       "clflushopt   960(%%r8)\n\t"
                       "clflushopt  1024(%%r8)\n\t"
                       "clflushopt  1088(%%r8)\n\t"
                       "clflushopt  1152(%%r8)\n\t"
                       "clflushopt  1216(%%r8)\n\t"
                       "clflushopt  1280(%%r8)\n\t"
                       "clflushopt  1344(%%r8)\n\t"
                       "clflushopt  1408(%%r8)\n\t"
                       "clflushopt  1472(%%r8)\n\t"
                       "clflushopt  1536(%%r8)\n\t"
                       "clflushopt  1600(%%r8)\n\t"
                       "clflushopt  1664(%%r8)\n\t"
                       "clflushopt  1728(%%r8)\n\t"
                       "clflushopt  1792(%%r8)\n\t"
                       "clflushopt  1856(%%r8)\n\t"
                       "clflushopt  1920(%%r8)\n\t"
                       "clflushopt  1984(%%r8)\n\t"
                       "addq $2048, %%r8\n\t"
                       "cmpq $0, %%r9\n\t"
                       "jg 1b\n\t"
                       : : "m"(i_buffer), "r"(i_length) : "r8","r9");
} 

int main(int argc, char* argv[]) {
  size_t l_n_bytes = 0;
  size_t l_n_levels = 0;
  size_t l_n_parts = 0;
  size_t l_n_workers = 0;
  size_t l_n_oiters = 0;
  size_t l_n_iiters = 0;
  char** l_n_buffers = 0;
  struct timeval l_startTime, l_endTime;
  struct timeval *l_startTime_arr, *l_endTime_arr;
  double *l_iterTimes;
  double l_avgtime;
  double l_mintime;
  double l_maxtime;
  double l_totalGiB, l_totalGiB_dram;
  size_t i, j, k;

#if defined(USE_CORE_PERF_L2IN) || defined(USE_CORE_PERF_SNP) || defined(USE_CORE_PERF_IPC)
  ctrs_core cc_a, cc_b, cc_s;
  zero_core_ctrs( &cc_a );
  zero_core_ctrs( &cc_b );
  zero_core_ctrs( &cc_s );

#if defined(USE_CORE_PERF_L2IN)
  setup_core_ctrs(CTRS_EXP_L2_BW);
#endif
#if defined(USE_CORE_PERF_SNP)
  setup_core_ctrs(CTRS_EXP_CORE_SNP_RSP);
#endif
#if defined(USE_CORE_PERF_IPC)
  setup_core_ctrs(CTRS_EXP_IPC);
#endif
#endif
#if defined(USE_UNCORE_PERF_DRAM_BW) || defined(USE_UNCORE_PERF_LLC_VICTIMS) || defined(USE_UNCORE_PERF_CHA_UTIL) || defined(USE_UNCORE_PREF_AK_UTIL) || defined(USE_UNCORE_PREF_IV_UTIL)
  ctrs_uncore uc_a, uc_b, uc_s;
  zero_uncore_ctrs( &uc_a );
  zero_uncore_ctrs( &uc_b );
  zero_uncore_ctrs( &uc_s );

#if defined(USE_UNCORE_PERF_DRAM_BW)
  setup_uncore_ctrs( CTRS_EXP_DRAM_CAS );
#endif
#if defined(USE_UNCORE_PERF_LLC_VICTIMS)
  setup_uncore_ctrs( CTRS_EXP_CHA_LLC_LOOKUP_VICTIMS );
#endif
#if defined(USE_UNCORE_PERF_CHA_UTIL)
  setup_uncore_ctrs( CTRS_EXP_CHA_UTIL );
#endif
#if defined(USE_UNCORE_PREF_AK_UTIL)
  setup_uncore_ctrs( CTRS_EXP_CMS_AK );
#endif
#if defined(USE_UNCORE_PREF_IV_UTIL)
  setup_uncore_ctrs( CTRS_EXP_CMS_IV );
#endif
#endif

  if ( argc != 7 ) {
    printf("Wrong parameters: ./app [# of 2K blocks] [#levels] [#partitions] [#workers] [#outer iterations] [#inner iterations]\n");
    return -1;
  }

  /* reading values from the command line */
  l_n_bytes = ((size_t)atoi(argv[1]))*2048;
  l_n_levels = atoi(argv[2]);
  l_n_parts = atoi(argv[3]);
  l_n_workers = atoi(argv[4]);
  l_n_oiters = atoi(argv[5]);
  l_n_iiters = atoi(argv[6]);

  /* validate the inputs */
  if ( (l_n_levels < 1) || (l_n_oiters < 1) || (l_n_iiters < 1) ) {
    printf("levels and iterations count needs to be non-zero and positive\n");
    return -1;
  }
  if ( ((l_n_bytes / l_n_parts) % 2048) != 0 ) {
    printf("each partition needs to be at least 2KB and the size of each partition needs to be a multipe of 2KB. ABORT!\n");
    return -1;
  }
  if ( (l_n_parts < 1) || ((l_n_workers % l_n_parts) != 0) ) {
    printf("paritions need to evenly divide workers. ABORT!\n");
    return -1;
  }

#ifdef __AVX512F__
  printf("using AVX512F\n");
#elif __AVX__
  printf("using AVX\n");
#elif __SSE2__
  printf("using SSE2\n");
#else
#error need at least SSE2
#endif

  /* allocating data */
  l_n_buffers = (char**) malloc( l_n_levels*sizeof(char**) );
  for ( i = 0; i < l_n_levels; ++i ) {
    posix_memalign( (void**)&(l_n_buffers[i]), 4096, l_n_bytes*sizeof(char) );
    memset( l_n_buffers[i], (int)i, l_n_bytes );
  }
  l_startTime_arr = (struct timeval*) malloc( l_n_oiters*sizeof(struct timeval) );
  l_endTime_arr = (struct timeval*) malloc( l_n_oiters*sizeof(struct timeval) );
  l_iterTimes = (double*) malloc( l_n_oiters*sizeof(double) );
  l_totalGiB = ((double)(l_n_bytes * l_n_levels * l_n_iiters * (l_n_workers / l_n_parts)))/(1024.0*1024.0*1024);
  l_totalGiB_dram = ((double)(l_n_bytes * l_n_levels))/(1024.0*1024.0*1024);

  printf("#Levels                                 : %lld\n", l_n_levels );
  printf("#iterations per level                   : %lld\n", l_n_iiters );
  printf("Buffer Size in GiB per level            : %f\n", (double)l_n_bytes/(1024.0*1024.0*1024.0));
  printf("Buffer Size in bytes per level          : %lld\n", l_n_bytes );
  printf("#workers used                           : %lld\n", l_n_workers );
  printf("#partition used                         : %lld\n", l_n_parts );
  printf("each worker reads #bytes                : %lld\n", (l_n_bytes / l_n_parts) );
  printf("each worker reads %% of buffer           : %f\n", (((double)(l_n_bytes / l_n_parts)) / ((double)l_n_bytes))*100.0 );
  printf("access indices per worker\n");
  for ( i = 0; i < l_n_workers; ++i ) {
    size_t my_size = l_n_bytes / l_n_parts;
    size_t my_offset = (size_t)i / ( l_n_workers / l_n_parts );
    printf("  worker %.3i                             : [ %lld , %lld [ ; length = %lld\n", i, my_offset * my_size, (my_offset * my_size) + my_size, my_size );
  }
  printf("Iteration Volume in GiB form LLC        : %f\n", l_totalGiB );
  printf("Iteration Volume in GiB from DRAM       : %f\n", l_totalGiB_dram );

  printf("warming up...\n");
  /* warming up */
#if defined(_OPENMP)
# pragma omp parallel private(i,j,k) num_threads(l_n_workers)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    for ( i = 0; i < 20; ++i ) {
      for ( j = 0; j < l_n_levels; ++j ) {
        for ( k = 0; k < l_n_iiters; ++k ) {
          char* my_buffer = l_n_buffers[j];
          size_t my_size = l_n_bytes / l_n_parts;
          size_t my_offset = (size_t)tid / ( l_n_workers / l_n_parts );
          size_t my_start = my_offset * my_size;
#if 1
          size_t my_shr_deg = l_n_workers / l_n_parts;
          size_t my_kern_size = my_size / my_shr_deg;
          size_t my_tid = tid % my_shr_deg;
          size_t l;
          for ( l = 0; l < my_shr_deg; ++l ) {
#if 0
            if (tid == 6) printf("tid: %lld, my_tid: %lld, my_shr_deg: %lld, my_size %lld, my_kern_size: %lld, my_start: %lld, inner_offset: %lld\n", tid, my_tid, my_shr_deg, my_size, my_kern_size, my_start, ( ( (l+my_tid) % my_shr_deg ) * my_kern_size) );
#endif
            read_buffer( my_buffer + my_start + ( ( (l+my_tid) % my_shr_deg ) * my_kern_size), my_kern_size );
          }
#else
          read_buffer( my_buffer + my_start, my_size );
#endif
        }
#if defined(_OPENMP)
# pragma omp barrier
#endif
      }
    }
  }
  printf("... done! [0][0] = %i\n", l_n_buffers[0][0] );  

  /* run the benchmark */
#if defined(USE_CORE_PERF_SNP) || defined(USE_CORE_PERF_L2IN) || defined(USE_CORE_PERF_IPC)
  read_core_ctrs( &cc_a );
#endif
#if defined(USE_UNCORE_PERF_DRAM_BW) || defined(USE_UNCORE_PERF_LLC_VICTIMS) || defined(USE_UNCORE_PERF_CHA_UTIL) || defined(USE_UNCORE_PREF_AK_UTIL) || defined(USE_UNCORE_PREF_IV_UTIL)
  read_uncore_ctrs( &uc_a );
#endif
  gettimeofday(&l_startTime, NULL);
#if defined(_OPENMP)
# pragma omp parallel private(i,j,k) num_threads(l_n_workers)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    for ( i = 0; i < l_n_oiters; ++i ) {
      if (tid == 0) gettimeofday(&(l_startTime_arr[i]), NULL);
      for ( j = 0; j < l_n_levels; ++j ) {
        for ( k = 0; k < l_n_iiters; ++k ) {
          char* my_buffer = l_n_buffers[j];
          size_t my_size = l_n_bytes / l_n_parts;
          size_t my_offset = (size_t)tid / ( l_n_workers / l_n_parts );
          size_t my_start = my_offset * my_size;
#if 1
          size_t my_shr_deg = l_n_workers / l_n_parts;
          size_t my_kern_size = my_size / my_shr_deg;
          size_t my_tid = tid % my_shr_deg;
          size_t l;
          for ( l = 0; l < my_shr_deg; ++l ) {
#if 0
            if (tid == 6) printf("tid: %lld, my_tid: %lld, my_shr_deg: %lld, my_size %lld, my_kern_size: %lld, my_start: %lld, inner_offset: %lld\n", tid, my_tid, my_shr_deg, my_size, my_kern_size, my_start, ( ( (l+my_tid) % my_shr_deg ) * my_kern_size) );
#endif
            read_buffer( my_buffer + my_start + ( ( (l+my_tid) % my_shr_deg ) * my_kern_size), my_kern_size );
          }
#else
          read_buffer( my_buffer + my_start, my_size );
#endif
        }
#if defined(_OPENMP)
# pragma omp barrier
#endif
      }
      if (tid == 0) gettimeofday(&(l_endTime_arr[i]), NULL);
    }
  }
  gettimeofday(&l_endTime, NULL);
#if defined(USE_CORE_PERF_SNP) || defined(USE_CORE_PERF_L2IN) || defined(USE_CORE_PERF_IPC)
  read_core_ctrs( &cc_b );
  difa_core_ctrs( &cc_a, &cc_b, &cc_s );
  divi_core_ctrs( &cc_s, l_n_oiters );
#endif
#if defined(USE_UNCORE_PERF_DRAM_BW) || defined(USE_UNCORE_PERF_LLC_VICTIMS) || defined(USE_UNCORE_PERF_CHA_UTIL) || defined(USE_UNCORE_PREF_AK_UTIL) || defined(USE_UNCORE_PREF_IV_UTIL)
  read_uncore_ctrs( &uc_b );
  difa_uncore_ctrs( &uc_a, &uc_b, &uc_s );
  divi_uncore_ctrs( &uc_s, l_n_oiters );
#endif
  l_mintime = FLT_MAX;
  l_maxtime = FLT_MIN;
  for ( i = 0; i < l_n_oiters; ++i ) {
    l_mintime = ( l_mintime > sec(l_startTime_arr[i], l_endTime_arr[i]) ) ? sec(l_startTime_arr[i], l_endTime_arr[i]) : l_mintime;
    l_maxtime = ( l_maxtime < sec(l_startTime_arr[i], l_endTime_arr[i]) ) ? sec(l_startTime_arr[i], l_endTime_arr[i]) : l_maxtime;
  }
  l_avgtime = sec(l_startTime, l_endTime)/((double)(l_n_oiters));
  printf("avg. Iteration Time in seconds          : %f\n", l_avgtime );
  printf("avg. Iteration L2in Bandwidth in GiB/s  : %f\n", l_totalGiB / l_avgtime ); 
  printf("avg. Iteration DRAM Bandwidth in GiB/s  : %f\n", l_totalGiB_dram / l_avgtime ); 
  printf("min. Iteration Time in seconds          : %f\n", l_mintime );
  printf("max. Iteration L2in Bandwidth in GiB/s  : %f\n", l_totalGiB / l_mintime ); 
  printf("max. Iteration DRAM Bandwidth in GiB/s  : %f\n", l_totalGiB_dram / l_mintime ); 
  printf("max. Iteration Time in seconds          : %f\n", l_maxtime );
  printf("min. Iteration L2in Bandwidth in GiB/s  : %f\n", l_totalGiB / l_maxtime ); 
  printf("min. Iteration DRAM Bandwidth in GiB/s  : %f\n", l_totalGiB_dram / l_maxtime ); 

#if defined(USE_CORE_PERF_L2IN)
  {
    bw_gibs bw_avg;
    get_l2_bw_core_ctrs( &cc_s, l_avgtime, &bw_avg );
    printf("AVG GiB/s (IN    L2): %f\n", bw_avg.rd);
    printf("AVG GiB/s (OUTS  L2): %f\n", bw_avg.wr);
    printf("AVG GiB/s (OUTNS L2): %f\n", bw_avg.wr2);
    printf("AVG GiB/s (DEM   L2): %f\n", bw_avg.wr3);
    printf("AVG GiB/s (DROP  L2): %f\n", bw_avg.wr4);
  }
#endif
#if defined(USE_CORE_PERF_SNP)
  {
    snp_rsp rsp;
    get_snp_rsp_core_ctrs( &cc_s, &rsp );
    printf("average #cycles per iteration : %f\n", rsp.cyc );
    printf("SNOOP RESP IHITI              : %f\n", rsp.ihiti );
    printf("SNOOP RESP IHITFSE            : %f\n", rsp.ihitfse );
    printf("SNOOP RESP IFWDM              : %f\n", rsp.ifwdm );
    printf("SNOOP RESP IFWDFE             : %f\n", rsp.ifwdfe );
    printf("avg SNOOP RESP IHITI / cycle  : %f\n", rsp.ihiti/rsp.cyc );
    printf("avg SNOOP RESP IHITFSE / cycle: %f\n", rsp.ihitfse/rsp.cyc );
    printf("avg SNOOP RESP IFWDM / cycle  : %f\n", rsp.ifwdm/rsp.cyc );
    printf("avg SNOOP RESP IFWDFE / cycle : %f\n", rsp.ifwdfe/rsp.cyc );
  }
#endif
#if defined(USE_CORE_PERF_IPC)
  {
    ipc_rate ipc;
    get_ipc_core_ctr( &cc_s, &ipc );
    printf("average #cycles per iteration : %f\n", ipc.cyc );
    printf("#intrs/core per iteration     : %f\n", ipc.instrs_core );
    printf("total #instr per iteration    : %f\n", ipc.instrs );
    printf("IPC per core                  : %f\n", ipc.ipc_core );
    printf("IPC per SOC                   : %f\n", ipc.ipc );
  }
#endif
#if defined(USE_UNCORE_PERF_DRAM_BW)
  {
    bw_gibs bw_avg;
    get_cas_ddr_bw_uncore_ctrs( &uc_s, l_avgtime, &bw_avg );
    printf("AVG GiB/s (IN  iMC): %f\n", bw_avg.rd);
    printf("AVG GiB/s (OUT iMC): %f\n", bw_avg.wr);
  }
#endif
#if defined(USE_UNCORE_PERF_LLC_VICTIMS)
  {
    llc_victims llc;
    get_llc_victim_bw_uncore_ctrs( &uc_s, l_avgtime, &llc );
    printf("LLC Lookup rd GiB/s : %f\n", llc.rd_bw );
    printf("LLC Lookup wr GiB/s : %f\n", llc.wr_bw );
    printf("LLC Victim E GiB/s  : %f\n", llc.bw_vic_e );
    printf("LLC Victim F GiB/s  : %f\n", llc.bw_vic_f );
    printf("LLC Victim M GiB/s  : %f\n", llc.bw_vic_m );
    printf("LLC Victim S GiB/s  : %f\n", llc.bw_vic_s );
    printf("LLC Victim E Count  : %f\n", llc.tot_vic_e );
    printf("LLC Victim F Count  : %f\n", llc.tot_vic_f );
    printf("LLC Victim M Count  : %f\n", llc.tot_vic_m );
    printf("LLC Victim S Count  : %f\n", llc.tot_vic_s );
  }
#endif
#if defined(USE_UNCORE_PERF_CHA_UTIL)
  {
    cha_util util;
    get_cha_util_uncore_ctrs( &uc_s, &util );
    printf("average #cycles per iteration : %f\n", util.cyc );
    printf("#intrs/cha per iteration      : %f\n", util.instrs_cha );
    printf("CHA Util                      : %f\n", util.util_cha );
  }
#endif
#if defined(USE_UNCORE_PREF_AK_UTIL)
  {
    int cha = 0;
    for ( cha = 0; cha < CTRS_NCHA; ++cha ) {
      printf("CHA %i: CMS cyc: %lld, AK_VERT: %lld, AK_HORZ: %lld\n", cha, uc_s.cms_clockticks[cha], uc_s.vert_ak_ring_in_use[cha], uc_s.horz_ak_ring_in_use[cha] );
    }
  }
#endif
#if defined(USE_UNCORE_PREF_IV_UTIL)
  {
    int cha = 0;
    for ( cha = 0; cha < CTRS_NCHA; ++cha ) {
      printf("CHA %i: CMS cyc: %lld, IV_VERT: %lld, IV_HORZ: %lld\n", cha, uc_s.cms_clockticks[cha], uc_s.vert_iv_ring_in_use[cha], uc_s.horz_iv_ring_in_use[cha] );
    }
  }
#endif
#if defined(USE_CORE_PERF_L2IN) && defined(USE_UNCORE_PERF_DRAM_BW)
  {
    cache_miss_rate mrate;
    get_l2_llc_misses_uncore_core_ctr( &cc_s, &uc_s, &mrate );
    printf("average #cycles per iteration  : %f\n", mrate.cyc );
    printf("average #instrs per iteration  : %f\n", mrate.instrs );
    printf("acc. #L2 misses per iteration  : %f\n", mrate.llc_rd_acc );
    printf("acc. #LLC misses per iteration : %f\n", mrate.dram_rd_acc );
    printf("L2 Miss Rate                   : %f\n", mrate.l2_miss_rate );
    printf("LLC Miss Rate                  : %f\n", mrate.llc_miss_rate );
  }
#endif

#if 0
#define FLUSH_CACHE_BEFORE
#endif
#if 0
#define FLUSH_CACHE_AFTER
#endif
#if 0
#define SEQ_LLC_ALLOC
#endif
#if 0
/* % n_parts -> n_shr_degree parallel cores */
#define GROUP_LLC_ALLOC_A
#endif
#if 0
/* % shr degree -> n_parts parallel cores */
#define GROUP_LLC_ALLOC_B
#endif


#if defined(SEQ_LLC_ALLOC) && defined(GROUP_LLC_ALLOC_A)
#error SEQ_LLC_ALLOC and GROUP_LLC_ALLOC_A cannot be defined at the same time
#endif
#if defined(SEQ_LLC_ALLOC) && defined(GROUP_LLC_ALLOC_B)
#error SEQ_LLC_ALLOC and GROUP_LLC_ALLOC_A cannot be defined at the same time
#endif
#if defined(GROUP_LLC_ALLOC_B) && defined(GROUP_LLC_ALLOC_A)
#error GROUP_LLC_ALLOC_B and GROUP_LLC_ALLOC_A cannot be defined at the same time
#endif
#if defined(FLUSH_CACHE_BEFORE) && defined(FLUSH_CACHE_AFTER)
#error FLUSH_CACHE_BEFORE and FLUSH_CACHE_AFTER cannot be defined at the same time
#endif

  printf("\nRunning detailed timing for round-robin read ...\n");
  {
    size_t* l_tsc_timer;
    size_t l_iter_to_analyze = 130;
    size_t l_level_to_analyze = 250;
    volatile size_t l_counter = 0;
    if ( l_iter_to_analyze >= l_n_oiters ) {
      printf(" iter to analyze is out of bounds!\n ");
      return -1;
    }
    if ( l_level_to_analyze >= l_n_levels ) {
      printf(" iter to analyze is out of bounds!\n ");
      return -1;
    }
    /* allocatopm pf timer arrary */
    l_tsc_timer = (size_t*) malloc( l_n_workers*l_n_levels*l_n_oiters*8*sizeof(size_t) );
    memset( (void*)l_tsc_timer, 0, l_n_workers*l_n_levels*l_n_oiters*8*sizeof(size_t) );
#if defined(_OPENMP)
# pragma omp parallel private(i,j,k) num_threads(l_n_workers)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    for ( i = 0; i < l_n_oiters; ++i ) {
      for ( j = 0; j < l_n_levels; ++j ) {
#if defined(FLUSH_CACHE_BEFORE)
        if ( ( i == l_iter_to_analyze ) && ( j == l_level_to_analyze ) ) {
          size_t t;
          for ( t = 0; t < l_level_to_analyze; ++t ) {
            char* my_wipe_buffer = l_n_buffers[t];
            size_t my_wipe_size = l_n_bytes / l_n_parts;
            size_t my_wipe_offset = (size_t)tid / ( l_n_workers / l_n_parts );
            size_t my_wipe_start = my_wipe_offset * my_wipe_size;
            clflush_buffer( my_wipe_buffer + my_wipe_start, my_wipe_size );
#if defined(_OPENMP)
# pragma omp barrier
#endif
          }
        }
#endif
        for ( k = 0; k < l_n_iiters; ++k ) {
          char* my_buffer = l_n_buffers[j];
          size_t my_size = l_n_bytes / l_n_parts;
          size_t my_offset = (size_t)tid / ( l_n_workers / l_n_parts );
          size_t my_start = my_offset * my_size;
          size_t my_shr_deg = l_n_workers / l_n_parts;
          size_t my_kern_size = my_size / my_shr_deg;
          size_t my_tid = tid % my_shr_deg;
          size_t l;
          l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 0] = __rdtsc(); 
          read_buffer( my_buffer + my_start + ( ( (0+my_tid) % my_shr_deg ) * my_kern_size), my_kern_size );
          l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 1] = __rdtsc();
#if defined(GROUP_LLC_ALLOC_A) ||  defined(GROUP_LLC_ALLOC_B) || defined(SEQ_LLC_ALLOC)
          if (tid == 0) l_counter = 0;
#endif
#if defined(_OPENMP)
# pragma omp barrier
#endif
#if defined(GROUP_LLC_ALLOC_A)
          if ( ( i == l_iter_to_analyze ) && ( j == l_level_to_analyze ) ) {
            while ( l_counter != (tid % l_n_parts) ) {
            }
            if ( my_shr_deg > 1 ) {
              l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 2] = __rdtsc(); 
              read_buffer( my_buffer + my_start + ( ( (1+my_tid) % my_shr_deg ) * my_kern_size), my_kern_size );
              l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 3] = __rdtsc(); 
            }
            if ( tid < l_n_parts ) l_counter++;
          } else {
            if ( my_shr_deg > 1 ) {
              l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 2] = __rdtsc();
              read_buffer( my_buffer + my_start + ( ( (1+my_tid) % my_shr_deg ) * my_kern_size), my_kern_size );
              l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 3] = __rdtsc();
            }
          }
#endif
#if defined(GROUP_LLC_ALLOC_B)
          if ( ( i == l_iter_to_analyze ) && ( j == l_level_to_analyze ) ) {
            while ( l_counter != (tid % my_shr_deg) ) {
            }
            if ( my_shr_deg > 1 ) {
              l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 2] = __rdtsc(); 
              read_buffer( my_buffer + my_start + ( ( (1+my_tid) % my_shr_deg ) * my_kern_size), my_kern_size );
              l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 3] = __rdtsc(); 
            }
            if ( tid < my_shr_deg ) l_counter++;
          } else {
            if ( my_shr_deg > 1 ) {
              l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 2] = __rdtsc();
              read_buffer( my_buffer + my_start + ( ( (1+my_tid) % my_shr_deg ) * my_kern_size), my_kern_size );
              l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 3] = __rdtsc();
            }
          }
#endif
#if defined(SEQ_LLC_ALLOC)
          if ( ( i == l_iter_to_analyze ) && ( j == l_level_to_analyze ) ) {
            while ( l_counter < tid ) {
            }
            if ( my_shr_deg > 1 ) {
              l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 2] = __rdtsc(); 
              read_buffer( my_buffer + my_start + ( ( (1+my_tid) % my_shr_deg ) * my_kern_size), my_kern_size );
              l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 3] = __rdtsc(); 
            }
            l_counter++;
          } else {
            if ( my_shr_deg > 1 ) {
              l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 2] = __rdtsc();
              read_buffer( my_buffer + my_start + ( ( (1+my_tid) % my_shr_deg ) * my_kern_size), my_kern_size );
              l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 3] = __rdtsc();
            }
          }
#endif
#if !defined(GROUP_LLC_ALLOC_A) && !defined(GROUP_LLC_ALLOC_B) && !defined(SEQ_LLC_ALLOC)
          if ( my_shr_deg > 1 ) {
            l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 2] = __rdtsc();
            read_buffer( my_buffer + my_start + ( ( (1+my_tid) % my_shr_deg ) * my_kern_size), my_kern_size );
            l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 3] = __rdtsc();
          }
#endif
#if defined(_OPENMP)
# pragma omp barrier
#endif
          l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 4] = __rdtsc(); 
          for ( l = 2; l < my_shr_deg; ++l ) {
            read_buffer( my_buffer + my_start + ( ( (l+my_tid) % my_shr_deg ) * my_kern_size), my_kern_size );
          }
          l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 5] = __rdtsc(); 
#if defined(_OPENMP)
# pragma omp barrier
#endif
#if defined(FLUSH_CACHE_AFTER)
          l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 6] = __rdtsc(); 
          clflush_buffer( my_buffer + my_start + ( ( (0+my_tid) % my_shr_deg ) * my_kern_size), my_kern_size );
          l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 7] = __rdtsc();
#if defined(_OPENMP)
# pragma omp barrier
#endif
#endif
        }
      }
    }
  }
    /* let's print the stats */
    {
      size_t l_tot_avg_cycles = 0;
      size_t l_tot_min_cycles = 0;
      size_t l_tot_max_cycles = 0;
      size_t l_avg_cycles = 0;
      size_t l_min_cycles = 0xffffffffffffffff;
      size_t l_max_cycles = 0;
      size_t my_size = l_n_bytes / l_n_parts;
      size_t my_shr_deg = l_n_workers / l_n_parts;
      size_t my_kern_size = my_size / my_shr_deg;
      size_t tid;
      j = l_level_to_analyze;
      i = l_iter_to_analyze;
          
      printf("\nLayer %lld and iteration %lld\n", j, i);
      printf("\nPhase I Perf - reading in data\n");
      printf("  per core:\n"); 
      for ( tid = 0; tid < l_n_workers; ++tid ) {
        size_t l_cycles = l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 1] - l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 0];
        l_avg_cycles += l_cycles;
        l_min_cycles = (l_cycles < l_min_cycles) ? l_cycles : l_min_cycles; 
        l_max_cycles = (l_cycles > l_max_cycles) ? l_cycles : l_max_cycles;
        printf("     worker %.3i: %f B/c (%lld, %lld, %lld, %lld) \n", tid, (double)my_kern_size/(double)l_cycles, my_kern_size, l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 0],  l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 1], l_cycles );
      }
      l_avg_cycles /= l_n_workers;
      printf("  avg: %f, min: %f, max: %f B/c\n  avg: %lld, min: %lld, max: %lld cycles\n  vol: %lld bytes\n", (double)my_kern_size/(double)l_avg_cycles, (double)my_kern_size/(double)l_max_cycles, (double)my_kern_size/(double)l_min_cycles,  l_avg_cycles, l_min_cycles, l_max_cycles, my_kern_size );

      l_tot_avg_cycles += l_avg_cycles;
      l_tot_min_cycles += l_min_cycles;
      l_tot_max_cycles += l_max_cycles;

      if ( my_shr_deg > 1 ) {
        l_avg_cycles = 0;
        l_min_cycles = 0xffffffffffffffff;
        l_max_cycles = 0;
        printf("\nPhase II Perf - making data shared\n");
        printf("  per core:\n"); 
        for ( tid = 0; tid < l_n_workers; ++tid ) {
          size_t l_cycles = l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 3] - l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 2];
          l_avg_cycles += l_cycles;
          l_min_cycles = (l_cycles < l_min_cycles) ? l_cycles : l_min_cycles; 
          l_max_cycles = (l_cycles > l_max_cycles) ? l_cycles : l_max_cycles;
          printf("     worker %.3i: %f B/c (%lld, %lld, %lld, %lld) \n", tid, (double)my_kern_size/(double)l_cycles, my_kern_size, l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 0],  l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 1], l_cycles );
        }
        l_avg_cycles /= l_n_workers;
        printf("  avg: %f, min: %f, max: %f B/c\n  avg: %lld, min: %lld, max: %lld cycles\n  vol: %lld bytes\n", (double)my_kern_size/(double)l_avg_cycles, (double)my_kern_size/(double)l_max_cycles, (double)my_kern_size/(double)l_min_cycles, l_avg_cycles, l_min_cycles, l_max_cycles, my_kern_size );

        l_tot_avg_cycles += l_avg_cycles;
        l_tot_min_cycles += l_min_cycles;
        l_tot_max_cycles += l_max_cycles;
      } 
      
      if ( my_shr_deg > 2 ) {
        l_avg_cycles = 0;
        l_min_cycles = 0xffffffffffffffff;
        l_max_cycles = 0;
        printf("\nPhase III Perf - reading shared data\n");
        printf("  per core:\n");
        size_t shared_size = my_size - (2*my_kern_size);
        for ( tid = 0; tid < l_n_workers; ++tid ) {
          size_t l_cycles = l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 5] - l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 4];
          l_avg_cycles += l_cycles;
          l_min_cycles = (l_cycles < l_min_cycles) ? l_cycles : l_min_cycles; 
          l_max_cycles = (l_cycles > l_max_cycles) ? l_cycles : l_max_cycles;
          printf("     worker %.3i: %f B/c (%lld, %lld, %lld, %lld) \n", tid, (double)shared_size/(double)l_cycles, shared_size, l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 0],  l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 1], l_cycles );
        }
        l_avg_cycles /= l_n_workers;
        printf("  avg: %f, min: %f, max: %f B/c\n  avg: %lld, min: %lld, max: %lld cycles\n  vol: %lld bytes\n", (double)shared_size/(double)l_avg_cycles, (double)shared_size/(double)l_max_cycles, (double)shared_size/(double)l_min_cycles, l_avg_cycles, l_min_cycles, l_max_cycles, shared_size );

        l_tot_avg_cycles += l_avg_cycles;
        l_tot_min_cycles += l_min_cycles;
        l_tot_max_cycles += l_max_cycles;
      }

#if defined(FLUSH_CACHE_AFTER)
      l_avg_cycles = 0;
      l_min_cycles = 0xffffffffffffffff;
      l_max_cycles = 0;
      printf("\nPhase IV Perf - flush caches\n");
      printf("  per core:\n"); 
      for ( tid = 0; tid < l_n_workers; ++tid ) {
        size_t l_cycles = l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 7] - l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 6];
        l_avg_cycles += l_cycles;
        l_min_cycles = (l_cycles < l_min_cycles) ? l_cycles : l_min_cycles; 
        l_max_cycles = (l_cycles > l_max_cycles) ? l_cycles : l_max_cycles;
        printf("     worker %.3i: %f B/c (%lld, %lld, %lld, %lld) \n", tid, (double)my_kern_size/(double)l_cycles, my_kern_size, l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 6],  l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 7], l_cycles );
        }
      l_avg_cycles /= l_n_workers;
      printf("  avg: %f, min: %f, max: %f B/c\n  avg: %lld, min: %lld, max: %lld cycles\n  vol: %lld bytes\n", (double)my_kern_size/(double)l_avg_cycles, (double)my_kern_size/(double)l_max_cycles, (double)my_kern_size/(double)l_min_cycles, l_avg_cycles, l_min_cycles, l_max_cycles, my_kern_size );

/*
      l_tot_avg_cycles += l_avg_cycles;
      l_tot_min_cycles += l_min_cycles;
      l_tot_max_cycles += l_max_cycles;
*/
#endif

      printf("\nTotal Perf - reading shared data\n");
      printf("  avg: %f, min: %f, max: %f B/c\n", (double)my_size/(double)l_tot_avg_cycles, (double)my_size/(double)l_tot_max_cycles, (double)my_size/(double)l_tot_min_cycles );
    }
    free( l_tsc_timer );
  }  
  /* free data */
  for ( i = 0; i < l_n_levels; ++i ) {
    free( l_n_buffers[i] );
  }
  free( l_n_buffers );
  free( l_startTime_arr );
  free( l_endTime_arr );
  free( l_iterTimes );

  return 0; 
}
