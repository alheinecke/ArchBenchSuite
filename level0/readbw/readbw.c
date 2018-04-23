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

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	10000000
#endif

#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES	10
#endif
#endif
#ifndef NTIMES
#   define NTIMES	10
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
  size_t l_n = 0;
  size_t l_i = 0;
  double* l_times;
  double l_result;
  double l_avgTime, l_minTime, l_maxTime;
  double l_size = (double)((size_t)STREAM_ARRAY_SIZE)*sizeof(double);
  double l_sum = (double)((size_t)STREAM_ARRAY_SIZE);
  struct timeval l_startTime, l_endTime;

  l_sum = ((l_sum*l_sum) + l_sum)/2;
  posix_memalign((void**)&l_data, 4096, ((size_t)STREAM_ARRAY_SIZE)*sizeof(double));
  l_times = (double*)malloc(sizeof(double)*NTIMES);

  printf("READ BW Test Size MiB: %f\n", (l_size/(1024.0*1024.0)));
  
  // init data
  #pragma omp parallel for
  for ( l_n = 0; l_n < STREAM_ARRAY_SIZE; l_n++ ) {
    l_data[l_n] = (double)l_n;
  }

  // run benchmark
  for( l_i = 0; l_i < NTIMES; l_i++ ) {
    l_result = 0.0;
    gettimeofday(&l_startTime, NULL);
    // we do manual reduction here as we don't rely on a smart OpenMP implementation
    #pragma omp parallel 
    {
      double l_res = 0.0;
      #pragma omp for
      for ( l_n = 0; l_n < STREAM_ARRAY_SIZE; l_n++ ) {
        l_res += l_data[l_n];
      }
      
      #pragma omp atomic
      l_result += l_res;
    }

    gettimeofday(&l_endTime, NULL);
    l_times[l_i] = sec(l_startTime, l_endTime);
  }

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
  printf("AVG MiB/s: %f\n", (l_size/(1024.0*1024.0))/l_avgTime);
  printf("MAX MiB/s: %f\n", (l_size/(1024.0*1024.0))/l_minTime);
  printf("MIN MiB/s: %f\n", (l_size/(1024.0*1024.0))/l_maxTime);

  if((l_result/l_sum)-1 < 1e-10) {
    printf("PASSED, %f\n", (l_result/l_sum)-1);
  } else {
    printf("FAILED, %f, %f, %f\n", l_sum, l_result, (l_result/l_sum)-1);
  }

  return 0; 
}
