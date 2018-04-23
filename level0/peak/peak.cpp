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

#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef BENCH_SLMSSE
#include "sse_float.hpp"
#include "sse_double.hpp"
#endif
#ifdef BENCH_SSE
#include "sse_float.hpp"
#include "sse_double.hpp"
#endif
#ifdef BENCH_AVX
#include "avx_float.hpp"
#include "avx_double.hpp"
#endif
#ifdef BENCH_AVX2
#include "avx_float.hpp"
#include "avx_double.hpp"
#endif
#ifdef BENCH_AVX512
#include "avx512_float.hpp"
#include "avx512_double.hpp"
#endif
#ifdef BENCH_ARMV8
#include "armv8_float.hpp"
#include "armv8_double.hpp"
#endif
#ifdef BENCH_POWER8
#include "vsx_float.hpp"
#include "vsx_double.hpp"
#endif

inline double sec(struct timeval start, struct timeval end) {
  return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))) / 1.0e6;
}

int main(int argc, char* argv[]) {
  double l_doubleDataAsm[8];
  float  l_floatDataAsm[16];
  struct timeval l_startTime, l_endTime;
  double l_time, l_flops;
  double l_numberOfThreads = 1;

  for (unsigned int i = 0; i < 8; i++) { 
    l_doubleDataAsm[i] = drand48();
  }
  for (unsigned int i = 0; i < 16; i++) { 
    l_floatDataAsm[i] = static_cast<float>(drand48());
  }

#ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp master
    l_numberOfThreads = static_cast<double>(omp_get_num_threads());
  }
#endif

#ifdef BENCH_SLMSSE
  std::cout << "Running on a x86_64 SSE (SLM)" << std::endl;
#endif
#ifdef BENCH_SSE
  std::cout << "Running on a x86_64 SSE" << std::endl;
#endif
#ifdef BENCH_AVX
  std::cout << "Running on a x86_64 AVX" << std::endl;
#endif
#ifdef BENCH_AVX2
  std::cout << "Running on a x86_64 AVX2" << std::endl;
#endif
#ifdef BENCH_AVX512
  std::cout << "Running on a x86_64 AVX512" << std::endl;
#endif
#ifdef BENCH_ARMV8
  std::cout << "Running on an ARMv8-A" << std::endl;
#endif
#ifdef BENCH_POWER8
  std::cout << "Running on a POWER8" << std::endl;
#endif
  std::cout << "Running with " << static_cast<int>(l_numberOfThreads) << " thread(s)" << std::endl;

  // single precision QFMA 
  gettimeofday(&l_startTime, NULL);
#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    gflops_float_qfma(l_floatDataAsm); 
  }
  gettimeofday(&l_endTime, NULL);
  l_time = sec(l_startTime, l_endTime);
  l_flops = 0.0;
#ifdef BENCH_AVX512_QFMA
  l_flops = 4.0*32.0*28.0*(100.0*100.0)*(100.0*100.0);
#endif
  l_flops *= l_numberOfThreads;
  std::cout << "GFLOPS float QFMA  " << (l_flops/1e9)/l_time << std::endl;

  // single precision FMA 
  gettimeofday(&l_startTime, NULL);
#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    gflops_float_fma(l_floatDataAsm); 
  }
  gettimeofday(&l_endTime, NULL);
  l_time = sec(l_startTime, l_endTime);
  l_flops = 0.0;
#ifdef BENCH_AVX2
  l_flops = 16.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX512
  l_flops = 32.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_ARMV8
  l_flops = 8.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_POWER8
  l_flops = 8.0*64.0*(100.0*100.0)*(100.0*100.0);
#endif
  l_flops *= l_numberOfThreads;
  std::cout << "GFLOPS float  FMA  " << (l_flops/1e9)/l_time << std::endl;

  // single precision MUL 
  gettimeofday(&l_startTime, NULL);
#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    gflops_float_mul(l_floatDataAsm); 
  }
  gettimeofday(&l_endTime, NULL);
  l_time = sec(l_startTime, l_endTime);
  l_flops = 0.0;
#ifdef BENCH_SLMSSE
  l_flops = 4.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_SSE
  l_flops = 4.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX
  l_flops = 8.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX2
  l_flops = 8.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX512
  l_flops = 16.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_ARMV8
  l_flops = 4.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_POWER8
  l_flops = 4.0*64.0*(100.0*100.0)*(100.0*100.0);
#endif
  l_flops *= l_numberOfThreads;
  std::cout << "GFLOPS float  MUL  " << (l_flops/1e9)/l_time << std::endl;

  // single precision ADD 
  gettimeofday(&l_startTime, NULL);
#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    gflops_float_add(l_floatDataAsm); 
  }
  gettimeofday(&l_endTime, NULL);
  l_time = sec(l_startTime, l_endTime);
  l_flops = 0.0;
#ifdef BENCH_SLMSSE
  l_flops = 4.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_SSE
  l_flops = 4.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX
  l_flops = 8.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX2
  l_flops = 8.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX512
  l_flops = 16.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_ARMV8
  l_flops = 4.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_POWER8
  l_flops = 4.0*64.0*(100.0*100.0)*(100.0*100.0);
#endif
  l_flops *= l_numberOfThreads;
  std::cout << "GFLOPS float  ADD  " << (l_flops/1e9)/l_time << std::endl;

  // single precision MADD 
  gettimeofday(&l_startTime, NULL);
#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    gflops_float_madd(l_floatDataAsm); 
  }
  gettimeofday(&l_endTime, NULL);
  l_time = sec(l_startTime, l_endTime);
  l_flops = 0.0;
#ifdef BENCH_SLMSSE
  l_flops = 4.0*15.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_SSE
  l_flops = 4.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX
  l_flops = 8.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX2
  l_flops = 8.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX512
  l_flops = 16.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_ARMV8
  l_flops = 4.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_POWER8
  l_flops = 4.0*64.0*(100.0*100.0)*(100.0*100.0);
#endif
  l_flops *= l_numberOfThreads;
  std::cout << "GFLOPS float  MADD " << (l_flops/1e9)/l_time << std::endl;
  
  // double precision FMA 
  gettimeofday(&l_startTime, NULL);
#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    gflops_double_fma(l_doubleDataAsm); 
  }
  gettimeofday(&l_endTime, NULL);
  l_time = sec(l_startTime, l_endTime);
  l_flops = 0.0;
#ifdef BENCH_AVX2
  l_flops = 8.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX512
  l_flops = 16.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_ARMV8
  l_flops = 4.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_POWER8
  l_flops = 4.0*64.0*(100.0*100.0)*(100.0*100.0);
#endif
  l_flops *= l_numberOfThreads;
  std::cout << "GFLOPS double FMA  " << (l_flops/1e9)/l_time << std::endl;

  // double precision MUL 
  gettimeofday(&l_startTime, NULL);
#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    gflops_double_mul(l_doubleDataAsm); 
  }
  gettimeofday(&l_endTime, NULL);
  l_time = sec(l_startTime, l_endTime);
  l_flops = 0.0;
#ifdef BENCH_SLMSSE
  l_flops = 2.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_SSE
  l_flops = 2.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX
  l_flops = 4.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX2
  l_flops = 4.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX512
  l_flops = 8.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_ARMV8
  l_flops = 2.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_POWER8
  l_flops = 2.0*64.0*(100.0*100.0)*(100.0*100.0);
#endif
  l_flops *= l_numberOfThreads;
  std::cout << "GFLOPS double MUL  " << (l_flops/1e9)/l_time << std::endl;

  // double precision ADD 
  gettimeofday(&l_startTime, NULL);
#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    gflops_double_add(l_doubleDataAsm); 
  }
  gettimeofday(&l_endTime, NULL);
  l_time = sec(l_startTime, l_endTime);
  l_flops = 0.0;
#ifdef BENCH_SLMSSE
  l_flops = 2.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_SSE
  l_flops = 2.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX
  l_flops = 4.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX2
  l_flops = 4.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX512
  l_flops = 8.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_ARMV8
  l_flops = 2.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_POWER8
  l_flops = 2.0*64.0*(100.0*100.0)*(100.0*100.0);
#endif
  l_flops *= l_numberOfThreads;
  std::cout << "GFLOPS double ADD  " << (l_flops/1e9)/l_time << std::endl;

  // double precision MADD 
  gettimeofday(&l_startTime, NULL);
#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    gflops_double_madd(l_doubleDataAsm);
  }
  gettimeofday(&l_endTime, NULL);
  l_time = sec(l_startTime, l_endTime);
  l_flops = 0.0;
#ifdef BENCH_SLMSSE
  l_flops = 2.0*15.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_SSE
  l_flops = 2.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX
  l_flops = 4.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX2
  l_flops = 4.0*16.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_AVX512
  l_flops = 8.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_ARMV8
  l_flops = 2.0*32.0*(100.0*100.0)*(100.0*100.0);
#endif
#ifdef BENCH_POWER8
  l_flops = 2.0*64.0*(100.0*100.0)*(100.0*100.0);
#endif
  l_flops *= l_numberOfThreads;
  std::cout << "GFLOPS double MADD " << (l_flops/1e9)/l_time << std::endl;

  return 0;
}

