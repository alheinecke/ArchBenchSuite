/******************************************************************************
** Copyright (c) 2013-2025, Alexander Heinecke, Ben Brock                    **
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
#include <cstdio>
#include <pmmintrin.h>
#include <sys/time.h>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "get_devices.hpp"

sycl::queue q;

using __half = sycl::half;
using __float2half = sycl::half;
using __nv_bfloat16 = oneapi::mkl::bfloat16;
using __float2bfloat16 = oneapi::mkl::bfloat16;

#define MEASURE_KERNEL

timeval begin;
timeval end;

void timer_start()
{
  gettimeofday(&begin,(struct timezone *)0);
}

double timer_stop()
{
  gettimeofday(&end,(struct timezone *)0);
  double seconds, useconds;
  double ret, tmp;

  if (end.tv_usec >= begin.tv_usec)
  {
    seconds = (double)end.tv_sec - (double)begin.tv_sec;
    useconds = (double)end.tv_usec - (double)begin.tv_usec;
  }
  else
  {
    seconds = (double)end.tv_sec - (double)begin.tv_sec;
    seconds -= 1;          // Correction
    useconds = (double)end.tv_usec - (double)begin.tv_usec;
    useconds += 1000000;      // Correction
  }

  // get time in seconds
  tmp = (double)useconds;
  ret = (double)seconds;
  tmp /= 1000000;
  ret += tmp;

  return ret;
}

// T x U -> V
// A is a matrix of type T
// B is a matrix of type U
// C is a matrix of type V
template <typename T, typename U, typename V>
void test_general(size_t m, size_t n, size_t k, size_t s, size_t r)
{
  // host-side allocation
  T* h_A = NULL;
  U* h_B = NULL;
  V* h_C = NULL;

  // Allocate memory on GFX
  T* d_A = NULL;
  U* d_B = NULL;
  V* d_C = NULL;

  double FLOPS = 0.0;
  double TIME = 0.0;

  std::cout << std::endl;

  for (size_t i = 0; i <= r; i ++)
  {
    FLOPS = 2.0*((double)m)*((double)n)*((double)k);

    // init data on host
    h_A = new T[m*k];
    h_B = new U[k*n];
    h_C = new V[m*n];
    for (size_t j = 0; j < m*k; j++) {
      h_A[j] = T((float)drand48());
    }
    for (size_t j = 0; j < k*n; j++) {
      h_B[j] = U((float)drand48());
    }
    for (size_t j = 0; j < m*n; j++) {
      h_C[j] = V(1.0);
    }

#ifndef MEASURE_KERNEL
    timer_start();
#endif

    d_A = sycl::malloc_device<T>(m*k, q);
    d_B = sycl::malloc_device<U>(k*n, q);
    d_C = sycl::malloc_device<V>(m*n, q);

    q.memcpy(d_A, h_A, m*k*sizeof(T));
    q.memcpy(d_B, h_B, k*n*sizeof(U));
    q.memcpy(d_C, h_C, m*n*sizeof(V));

    V alpha = V(1.0);
    V beta  = V(0.0);
#ifdef MEASURE_KERNEL
    timer_start();
#endif
    for ( int z = 0; z < s; ++z ) {
      /*
      cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
      */

      oneapi::mkl::blas::column_major::gemm(q,
                            oneapi::mkl::transpose::nontrans,
                            oneapi::mkl::transpose::nontrans,
                            m, n, k,
                            alpha,
                            d_A, m,
                            d_B, k,
                            beta,
                            d_C, m);
    }
    q.wait();
#ifdef MEASURE_KERNEL
    TIME = timer_stop();
#endif

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);

#ifndef MEASURE_KERNEL
    TIME = timer_stop();
#endif

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    // Print results
    std::cout << i << ";" << m << ";" << n << ";" << k << ";" << TIME/((double)s) << ";" << (FLOPS/(1e9))/(TIME/(double)s) << std::endl;
  }

  std::cout << std::endl;
}

void test_half_half_general(size_t m, size_t n, size_t k, size_t s, size_t r)
{
  test_general<__half, __half, __half>(m, n, k, s, r);
}

void test_half_float_general(size_t m, size_t n, size_t k, size_t s, size_t r)
{
  test_general<__half, __half, float>(m, n, k, s, r);
}

void test_bfloat_float_general(size_t m, size_t n, size_t k, size_t s, size_t r)
{
  test_general<__nv_bfloat16, __nv_bfloat16, float>(m, n, k, s, r);
}

void test_bfloat_bfloat_general(size_t m, size_t n, size_t k, size_t s, size_t r)
{
  test_general<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(m, n, k, s, r);
}

void test_float_general(size_t m, size_t n, size_t k, size_t s, size_t r, char use_tf32)
{
  test_general<float, float, float>(m, n, k, s, r);
}


void test_double_general(size_t m, size_t n, size_t k, size_t s, size_t r)
{
  test_general<double, double, double>(m, n, k, s, r);
}

void write_help_general()
{
  std::cout << std::endl << "Wrong parameters! Please use:" << std::endl;
  std::cout << "  device number" << std::endl;
  std::cout << "  0: fp32; 1: fp64; 2: fp16 with fp16-acc; 3: fp16 with fp32-acc; 4: tf32; 5: bf16 wtih fp32-acc, fp32 out; 6: bf16 wtih fp32-acc, bf16 out" << std::endl;
  std::cout << "  m" << std::endl;
  std::cout << "  n" << std::endl;
  std::cout << "  k" << std::endl;
  std::cout << "  s" << std::endl;
  std::cout << "  r" << std::endl << std::endl;
  std::cout << "Example: 0 0 4096 448 2048 100 10" << std::endl << std::endl;
}

int main(int argc, char* argv[])
{
  if (argc != 8)
  {
    write_help_general();
    return 0;
  }

  size_t dev = atoi(argv[1]);
  size_t mode = atoi(argv[2]);
  size_t m = atoi(argv[3]);
  size_t n = atoi(argv[4]);
  size_t k = atoi(argv[5]);
  size_t s = atoi(argv[6]);
  size_t r = atoi(argv[7]);

  // Change to `helper::get_devices(sycl::default_selector_v, false)` to use
  // whole GPU instead of a single tile on PVC.  (i.e. "implicit scaling")
  auto devices = helper::get_devices(sycl::default_selector_v);
  sycl::context context(devices);

  auto device = devices[dev % devices.size()];
  q = sycl::queue(context, device, {sycl::property::queue::in_order()});

  srand48(567);

  if (mode == 0)
  {
    test_float_general(m, n, k, s, r, 0);
  }
  else if (mode == 1)
  {
    test_double_general(m, n, k, s, r);
  }
  else if (mode == 2)
  {
    test_half_half_general(m, n, k, s, r);
  }
  else if (mode == 3)
  {
    test_half_float_general(m, n, k, s, r);
  }
  else if (mode == 4)
  {
    test_float_general(m, n, k, s, r, 1);
  }
  else if (mode == 5)
  {
    test_bfloat_float_general(m, n, k, s, r);
  }
  else if (mode == 6)
  {
    test_bfloat_bfloat_general(m, n, k, s, r);
  }
  else
  {
    write_help_general();
    return 1;
  }

  return 0;
}
