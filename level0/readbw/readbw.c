/******************************************************************************
** Copyright (c) 2013-2018, Alexander Heinecke                               **
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
#define USE_UNCORE_PERF_COUNTERS
#if 0
#define USE_DRAM_COUNTERS
#endif
#endif
#if 0
#define USE_CORE_PERF_COUNTERS
#endif

#if __AVX512F__
#include <immintrin.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#if defined(USE_UNCORE_PERF_COUNTERS) || defined(USE_CORE_PERF_COUNTERS)
#include "./../common/perf_counter_markers.h"
#endif

#if 0
#define USE_THREE_BUFFERS
#endif

#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	1000000
#endif

#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES	10000
#endif
#endif
#ifndef NTIMES
#   define NTIMES	10000
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

int main(int argc, char* argv[]) {
  double* l_data;
#ifdef USE_THREE_BUFFERS
  double* l_data2;
  double* l_data3;
#endif
  size_t l_n = 0;
  size_t l_i = 0;
  double* l_times;
  double l_result;
  double l_avgTime, l_minTime, l_maxTime;
  double l_size = (double)((size_t)STREAM_ARRAY_SIZE)*sizeof(double);
  double l_sum = (double)((size_t)STREAM_ARRAY_SIZE);
  struct timeval l_startTime, l_endTime;
#ifdef USE_UNCORE_PERF_COUNTERS
  ctrs_uncore a, b, s;
  bw_gibs bw_min, bw_max, bw_avg;
  llc_victims llc_vic_min, llc_vic_max, llc_vic_avg;

#ifdef USE_DRAM_COUNTERS
  setup_uncore_ctrs( CTRS_EXP_DRAM_CAS );
#else
  setup_uncore_ctrs( CTRS_EXP_CHA_LLC_LOOKUP_VICTIMS );
#endif
  zero_uncore_ctrs( &a );
  zero_uncore_ctrs( &b );
  zero_uncore_ctrs( &s );
#endif
#ifdef USE_CORE_PERF_COUNTERS
  ctrs_core a, b, s;
  bw_gibs bw_min, bw_max, bw_avg;

  setup_core_ctrs( CTRS_EXP_L2_BW );
  zero_core_ctrs( &a );
  zero_core_ctrs( &b );
  zero_core_ctrs( &s );
#endif

  l_sum = ((l_sum*l_sum) + l_sum)/2;
#ifdef USE_THREE_BUFFERS
  l_sum *= 3;
#endif
  posix_memalign((void**)&l_data,  2097152, ((size_t)STREAM_ARRAY_SIZE)*sizeof(double));
#ifdef USE_THREE_BUFFERS
  posix_memalign((void**)&l_data2, 2097152, ((size_t)STREAM_ARRAY_SIZE)*sizeof(double));
  posix_memalign((void**)&l_data3, 2097152, ((size_t)STREAM_ARRAY_SIZE)*sizeof(double));
#endif
  l_times = (double*)malloc(sizeof(double)*NTIMES);

#ifndef USE_THREE_BUFFERS
  printf("READ BW Test Size MiB: %f\n", (l_size/(1024.0*1024.0)));
#else
  printf("READ BW Test Size MiB: %f\n", (l_size*3/(1024.0*1024.0)));
#endif

  // init data
  #pragma omp parallel for
  for ( l_n = 0; l_n < STREAM_ARRAY_SIZE; l_n++ ) {
    l_data[l_n] = (float)l_n;
#ifdef USE_THREE_BUFFERS
    l_data2[l_n] = (float)l_n;
    l_data3[l_n] = (float)l_n;
#endif
  }

#ifdef USE_UNCORE_PERF_COUNTERS
  read_uncore_ctrs( &a );
#endif
#ifdef USE_CORE_PERF_COUNTERS
  read_core_ctrs( &a );
#endif
  // run benchmark
  for( l_i = 0; l_i < NTIMES; l_i++ ) {
    l_result = 0.0;
    gettimeofday(&l_startTime, NULL);
    // we do manual reduction here as we don't rely on a smart OpenMP implementation
    #pragma omp parallel 
    {
      double l_res = 0.0;

#ifdef __AVX512F__
      __m512d l_acc = _mm512_setzero_pd();
#ifdef USE_THREE_BUFFERS
      __m512d l_acc2 = _mm512_setzero_pd();
      __m512d l_acc3 = _mm512_setzero_pd();
#endif
#endif
      #pragma omp for nowait
      for ( l_n = 0; l_n < STREAM_ARRAY_SIZE; l_n+=8 ) {
#ifdef __AVX512F__
        l_acc = _mm512_add_pd( l_acc, _mm512_load_pd( l_data + l_n ));
#ifdef USE_THREE_BUFFERS
        l_acc2 = _mm512_add_pd( l_acc2, _mm512_load_pd( l_data2 + l_n ));
      	l_acc3 = _mm512_add_pd( l_acc3, _mm512_load_pd( l_data3 + l_n ));
#endif
#else
      	l_res += l_data[l_n];
#endif
      }
#ifdef __AVX512F__
      l_res = _mm512_reduce_add_pd( l_acc );
#ifdef USE_THREE_BUFFERS
      l_res += _mm512_reduce_add_pd( l_acc2 );
      l_res += _mm512_reduce_add_pd( l_acc3 );
#endif
#endif
      #pragma omp atomic
      l_result += l_res;
    }
    gettimeofday(&l_endTime, NULL);
    l_times[l_i] = sec(l_startTime, l_endTime);
  }
#ifdef USE_UNCORE_PERF_COUNTERS
  read_uncore_ctrs( &b );
  difa_uncore_ctrs( &a, &b, &s );
  divi_uncore_ctrs( &s, NTIMES );
#endif
#ifdef USE_CORE_PERF_COUNTERS
  read_core_ctrs( &b );
  difa_core_ctrs( &a, &b, &s );
  divi_core_ctrs( &s, NTIMES );
#endif
  
  // postprocess timing
  l_avgTime = 0.0;
  l_minTime = 100000.0;
  l_maxTime = 0.0;
  for( l_i = 0; l_i < NTIMES; l_i++ ) {
    l_avgTime += l_times[l_i];
    l_minTime = MIN(l_minTime, l_times[l_i]);
    l_maxTime = MAX(l_maxTime, l_times[l_i]);
  }
  l_avgTime /= (double)NTIMES;
  
  // output
#ifndef USE_THREE_BUFFERS
  printf("AVG GiB/s  (calculated): %f\n", (l_size/(1024.0*1024.0*1024.0))/l_avgTime);
  printf("MAX GiB/s  (calculated): %f\n", (l_size/(1024.0*1024.0*1024.0))/l_minTime);
  printf("MIN GiB/s  (calculated): %f\n", (l_size/(1024.0*1024.0*1024.0))/l_maxTime);
#else
  printf("AVG GiB/s  (calculated): %f\n", (l_size*3.0/(1024.0*1024.0*1024.0))/l_avgTime);
  printf("MAX GiB/s  (calculated): %f\n", (l_size*3.0/(1024.0*1024.0*1024.0))/l_minTime);
  printf("MIN GiB/s  (calculated): %f\n", (l_size*3.0/(1024.0*1024.0*1024.0))/l_maxTime);
#endif
#ifdef USE_UNCORE_PERF_COUNTERS
#ifdef USE_DRAM_COUNTERS
  get_cas_ddr_bw_uncore_ctrs( &s, l_maxTime, &bw_min );
  get_cas_ddr_bw_uncore_ctrs( &s, l_minTime, &bw_max );
  get_cas_ddr_bw_uncore_ctrs( &s, l_avgTime, &bw_avg );
  printf("AVG GiB/s (uncore ctrs): %f\n", bw_avg.rd);
  printf("MAX GiB/s (uncore ctrs): %f\n", bw_max.rd);
  printf("MIN GiB/s (uncore ctrs): %f\n", bw_min.rd);
#else
  get_llc_victim_bw_uncore_ctrs( &s, l_maxTime, &llc_vic_max );
  get_llc_victim_bw_uncore_ctrs( &s, l_minTime, &llc_vic_min );
  get_llc_victim_bw_uncore_ctrs( &s, l_avgTime, &llc_vic_avg );
  printf("AVG GiB/s (uncore ctrs): %f\n", llc_vic_avg.rd_bw);
  printf("MAX GiB/s (uncore ctrs): %f\n", llc_vic_max.rd_bw);
  printf("MIN GiB/s (uncore ctrs): %f\n", llc_vic_min.rd_bw);
#endif
#endif
#ifdef USE_CORE_PERF_COUNTERS
  get_l2_bw_core_ctrs( &s, l_maxTime, &bw_min );
  get_l2_bw_core_ctrs( &s, l_minTime, &bw_max );
  get_l2_bw_core_ctrs( &s, l_avgTime, &bw_avg );
  printf("AVG GiB/s (IN    L2): %f\n", bw_avg.rd);
  printf("MAX GiB/s (IN    L2): %f\n", bw_max.rd);
  printf("MIN GiB/s (IN    L2): %f\n", bw_min.rd);
  printf("AVG GiB/s (OUTS  L2): %f\n", bw_avg.wr);
  printf("MAX GiB/s (OUTS  L2): %f\n", bw_max.wr);
  printf("MIN GiB/s (OUTS  L2): %f\n", bw_min.wr);
  printf("AVG GiB/s (OUTNS L2): %f\n", bw_avg.wr2);
  printf("MAX GiB/s (OUTNS L2): %f\n", bw_max.wr2);
  printf("MIN GiB/s (OUTNS L2): %f\n", bw_min.wr2);
  printf("AVG GiB/s (DEM   L2): %f\n", bw_avg.wr3);
  printf("MAX GiB/s (DEM   L2): %f\n", bw_max.wr3);
  printf("MIN GiB/s (DEM   L2): %f\n", bw_min.wr3);
  printf("AVG GiB/s (DROP  L2): %f\n", bw_avg.wr4);
  printf("MAX GiB/s (DROP  L2): %f\n", bw_max.wr4);
  printf("MIN GiB/s (DROP  L2): %f\n", bw_min.wr4);
#endif

  if((l_result/l_sum)-1 < 1e-10) {
    printf("PASSED, %f\n", (l_result/l_sum)-1);
  } else {
    printf("FAILED, %f, %f, %f\n", l_sum, l_result, (l_result/l_sum)-1);
  }

  return 0; 
}
