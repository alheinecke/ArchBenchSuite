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

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#ifdef __MKL
#include <mkl.h>
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
  if (argc != 9) {
    printf(" M N K trA trB rep1 rep2 pack\n");
    return -1;
  }

  MKL_INT m   = atoi(argv[1]);
  MKL_INT n   = atoi(argv[2]);
  MKL_INT k   = atoi(argv[3]);
  char   trA  = *(argv[4]);
  char   trB  = *(argv[5]);
  size_t rep1 = atoi(argv[6]);
  size_t rep2 = atoi(argv[7]);
  int    pack = atoi(argv[8]);

  size_t i;
  size_t r1;
  size_t r2;
  struct timeval start, end;
  float alpha = 1.0f;
  float beta = 1.0f;
  char  matA = 'A';
  char  matB = 'B';
  char  trP  = 'P';
  float* A  = (float*)_mm_malloc((size_t)m*k*sizeof(float), 2097152);
  float* B  = (float*)_mm_malloc((size_t)k*n*sizeof(float), 2097152);
  float* C  = (float*)_mm_malloc((size_t)m*n*sizeof(float), 2097152);
#ifdef __MKL
  float* Ap = sgemm_alloc(&matA, &m, &n, &k);
  float* Bp = sgemm_alloc(&matB, &m, &n, &k);
#endif
  char host[256];

  trA = 'N';
  trB = 'N';

  for (i = 0; i < ((size_t)m*n); i++)
    C[i] = 0.0f;
    
  for (i = 0; i < ((size_t)m*k); i++)
    A[i] = (float)i;
 
  for (i = 0; i < ((size_t)k*n); i++)
    B[i] = (float)i;
    
  sgemm_(&trA, &trB, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);

  printf("host,m,n,k,transA,transB,reps,time,flops,GFLOPS\n");
    
  for (r2 = 0; r2 < rep2; ++r2) {
    gettimeofday(&start, NULL); 
#ifdef __MKL
    if ( pack == 1 || pack == 3 ) {
      /* let's pack A matrix */
      sgemm_pack(&matA, &trA, &m, &n, &k, &alpha, A, &m, Ap);    
    }
    if ( pack == 2 || pack == 3 ) {
      /* let's pack B matrix */
      sgemm_pack(&matB, &trB, &m, &n, &k, &beta, B, &k, Bp);    
    }
    if ( pack == 0 ) {
      /* run packed sgemm */
      for (r1 = 0; r1 < rep1; ++r1) {
        sgemm_(&trA, &trB, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
      }
    } else if ( pack == 1 ) {
      for (r1 = 0; r1 < rep1; ++r1) {
        sgemm_compute(&trP, &trB, &m, &n, &k, Ap, &m, B, &k, &beta, C, &m);
      }
    } else if ( pack == 2 ) {
      for (r1 = 0; r1 < rep1; ++r1) {
        sgemm_compute(&trA, &trP, &m, &n, &k, A, &m, Bp, &k, &beta, C, &m);
      }
    } else if ( pack == 3 ) {
      for (r1 = 0; r1 < rep1; ++r1) {
        sgemm_compute(&trP, &trP, &m, &n, &k, Ap, &m, Bp, &k, &beta, C, &m);
      }
    }
#else
    for (r1 = 0; r1 < rep1; ++r1) { 
      sgemm_(&trA, &trB, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
    }
#endif    
    gettimeofday(&end, NULL);

    double time = sec(start, end);
    double dm = (double)m;
    double dn = (double)n;
    double dk = (double)k;
    double dr = (double)rep1;
    gethostname(host, 255);
    printf("%s,%i,%i,%i,%c,%c,%lld,%f,%f,%f\n", host, m, n, k, trA, trB, rep1, time/dr, dk*dm*dn*2.0, ((dr*dk*dm*dn*2.0)/1e9)/time );
  }

  _mm_free( A );
  _mm_free( B );
  _mm_free( C );
#ifdef __MKL
  sgemm_free( Ap );
  sgemm_free( Bp );
#endif

  return 0;
}
