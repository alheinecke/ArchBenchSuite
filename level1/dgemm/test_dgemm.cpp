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
#include <assert.h>
#ifdef __MKL
#include <mkl.h>
#include <mkl_cblas.h>
#else
#include <acml.h>
#endif
#include <immintrin.h>
#include <sys/time.h>
#include <unistd.h>

inline double sec(struct timeval start, struct timeval end)
{
    return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec))))/1.0e6;
}

int main(int argc, char* argv[])
{
    if (argc != 8)
    {
        std::cout << " M N K trA trB rep1 rep2" << std::endl;
        return -1;
    }

    size_t m = atoi(argv[1]);
    size_t n = atoi(argv[2]);
    size_t k = atoi(argv[3]);
    char   trA = *(argv[4]);
    char   trB = *(argv[5]);
    size_t rep1 = atoi(argv[6]);
    size_t rep2 = atoi(argv[7]);

    double* A = (double*)_mm_malloc(m*k*sizeof(double), 2097152);
    double* B = (double*)_mm_malloc(k*n*sizeof(double), 2097152);
    double* C = (double*)_mm_malloc(m*n*sizeof(double), 2097152);
    char host[256];
    trA = 'N';
    trB = 'N';
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    for (size_t i = 0; i < (m*n); i++)
        C[i] = 0.0f;
    
    for (size_t i = 0; i < (m*k); i++)
        A[i] = (double)i;
 
    for (size_t i = 0; i < (k*n); i++)
        B[i] = (double)i;
    
    cblas_dgemm(CblasColMajor, transA, transB, (const int)m, 
         (const int)n, (const int)k, 1.0, (const double*)A,
         (const int)m, (const double*)B, (const int)k, 0.0, C, (const int)m);
     
    struct timeval start, end;

    std::cout << "host,m,n,k,transA,transB,reps,time,flops,GFLOPS" << std::endl;
    
    for (int r2 = 0; r2 < rep2; ++r2) {
      gettimeofday(&start, NULL);
   
#ifdef __MKL

      for (int r1 = 0; r1 < rep1; ++r1) {
        //cblas_dgemm(CblasColMajor, transA, transB, (const int)m, 
        //   (const int)n, (const int)k, 1.0, (const double*)A,
        //   (const int)m, (const double*)B, (const int)k, 0.0, C, (const int)m);
        cblas_dgemm(CblasColMajor, transA, transB, (const int)m, 
             (const int)n, (const int)k, 1.0, (const double*)A,
             (const int)m, (const double*)B, (const int)k, 0.0, (double*)C, (const int)m);
      }
#else
      double one = 1.0;
      double zero = 0.0;
      for (int r1 = 0; r1 < rep1; ++r1) { 
//      dgemm(&trA, &trB, &m, 
//           &n, &k, &one, (double*)A,
//           &m, (double*)B, &k, &zero, C, &m);
        dgemm(trA, trB, m, 
             n, k, one, (double*)A,
             m, (double*)B, k, zero, C, m);
      }
#endif    
      gettimeofday(&end, NULL);

      double time = sec(start, end);
      double dm = (double)m;
      double dn = (double)n;
      double dk = (double)k;
      double dr = (double)rep1;
      gethostname(host, 255);
      std::cout << host <<  "," << m << "," << n << "," << k << "," << trA << "," << trB << "," << rep1 << "," << time/dr << "," << dk*dn*dm*2.0 << "," << ((dr*dk*dn*dm*2.0)/1e9)/time << std::endl;
    }

    return 0;
}
