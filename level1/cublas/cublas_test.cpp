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

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <pmmintrin.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

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
		seconds -= 1;					// Correction
		useconds = (double)end.tv_usec - (double)begin.tv_usec;
		useconds += 1000000;			// Correction
	}

	// get time in seconds
	tmp = (double)useconds;
	ret = (double)seconds;
	tmp /= 1000000;
	ret += tmp;

	return ret;
}

void test_half_half_general(size_t m, size_t n, size_t k, size_t s, size_t r)
{
	// host-side allocation
	__half* h_A = NULL;
	__half* h_B = NULL;
	__half* h_C = NULL;

	// Allocate memory on GFX
	__half* d_A = NULL;
	__half* d_B = NULL;
	__half* d_C = NULL;
	
	double FLOPS = 0.0;
	double TIME = 0.0;
	
  cudaError_t cuerror;
	cublasStatus_t status;
  cublasHandle_t handle; 

  status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS initialization error\n");
	}

	status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH );  /* CUBLAS_DEFAULT_MATH */	
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS set MathMode error\n");
	}

	std::cout << std::endl;

	for (size_t i = 0; i <= r; i ++)
	{
		FLOPS = 2.0*((double)m)*((double)n)*((double)k);
		
		// init data on host
		h_A = new __half[m*k];
		h_B = new __half[k*n];
		h_C = new __half[m*n];
		for (size_t j = 0; j < i*i; j++)
		{
			h_A[j] = __float2half((float)drand48());
			h_B[j] = __float2half((float)drand48());
			h_C[j] = __float2half(1.0);
		}
		
#ifndef MEASURE_KERNEL
		timer_start();
#endif
		
		cuerror = cudaMalloc((void**)&d_A, m*k*sizeof(__half));
		if (cuerror != cudaSuccess) {
		printf ("!!!! device memory allocation error (A)\n");
		}

		cuerror = cudaMalloc((void**)&d_B, k*n*sizeof(__half));
		if (cuerror != cudaSuccess) {
		printf ("!!!! device memory allocation error (B)\n");
		}

		cuerror = cudaMalloc((void**)&d_C, m*n*sizeof(__half));
		if (cuerror != cudaSuccess) {
			printf ("!!!! device memory allocation error (C)\n");
		}

		status = cublasSetVector(m*k, sizeof(__half), h_A, 1, d_A, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write A)\n");
		}

		status = cublasSetVector(k*n, sizeof(__half), h_B, 1, d_B, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write B)\n");
		}
		status = cublasSetVector(m*n, sizeof(__half), h_C, 1, d_C, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write C)\n");
		}
    __half alpha = __float2half(1.0);
		__half beta  = __float2half(0.0);
#ifdef MEASURE_KERNEL
		timer_start();
#endif
    for ( int z = 0; z < s; ++z ) {
		  cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    }
    cudaDeviceSynchronize();
#ifdef MEASURE_KERNEL
		TIME = timer_stop();
#endif

		cuerror = cudaFree(d_A);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (A)\n");
		}

		cuerror = cudaFree(d_B);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (B)\n");
		}

		cuerror = cudaFree(d_C);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (C)\n");
		}
		
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
	
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! shutdown error\n");
	}
}

void test_half_float_general(size_t m, size_t n, size_t k, size_t s, size_t r)
{
	// host-side allocation
	__half* h_A = NULL;
	__half* h_B = NULL;
  float* h_C = NULL;

	// Allocate memory on GFX
	__half* d_A = NULL;
	__half* d_B = NULL;
	float* d_C = NULL;
	
	double FLOPS = 0.0;
	double TIME = 0.0;
	
  cudaError_t cuerror;
	cublasStatus_t status;
  cublasHandle_t handle; 

  status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS initialization error\n");
	}

	status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH );  /* CUBLAS_DEFAULT_MATH */	
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS set MathMode error\n");
	}

	std::cout << std::endl;

	for (size_t i = 0; i <= r; i ++)
	{
		FLOPS = 2.0*((double)m)*((double)n)*((double)k);
		
		// init data on host
		h_A = new __half[m*k];
		h_B = new __half[k*n];
		h_C = new float[m*n];
		for (size_t j = 0; j < i*i; j++)
		{
			h_A[j] = __float2half((float)drand48());
			h_B[j] = __float2half((float)drand48());
			h_C[j] = 1.0;
		}
		
#ifndef MEASURE_KERNEL
		timer_start();
#endif
		
		cuerror = cudaMalloc((void**)&d_A, m*k*sizeof(__half));
		if (cuerror != cudaSuccess) {
		printf ("!!!! device memory allocation error (A)\n");
		}

		cuerror = cudaMalloc((void**)&d_B, k*n*sizeof(__half));
		if (cuerror != cudaSuccess) {
		printf ("!!!! device memory allocation error (B)\n");
		}

		cuerror = cudaMalloc((void**)&d_C, m*n*sizeof(float));
		if (cuerror != cudaSuccess) {
			printf ("!!!! device memory allocation error (C)\n");
		}

		status = cublasSetVector(m*k, sizeof(__half), h_A, 1, d_A, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write A)\n");
		}

		status = cublasSetVector(k*n, sizeof(__half), h_B, 1, d_B, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write B)\n");
		}

    status = cublasSetVector(m*n, sizeof(float), h_C, 1, d_C, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write C)\n");
		}
    __half alpha = __float2half(1.0);
		float beta  = 1.0;
#ifdef MEASURE_KERNEL
		timer_start();
#endif
    for ( int z = 0; z < s; ++z ) {
		  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, CUDA_R_16F, m, d_B, CUDA_R_16F, k, &beta, d_C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();
#ifdef MEASURE_KERNEL
		TIME = timer_stop();
#endif

		cuerror = cudaFree(d_A);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (A)\n");
		}

		cuerror = cudaFree(d_B);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (B)\n");
		}

		cuerror = cudaFree(d_C);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (C)\n");
		}
		
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
	
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! shutdown error\n");
	}
}

void test_bfloat_float_general(size_t m, size_t n, size_t k, size_t s, size_t r)
{
	// host-side allocation
	__nv_bfloat16* h_A = NULL;
	__nv_bfloat16* h_B = NULL;
  float* h_C = NULL;

	// Allocate memory on GFX
	__nv_bfloat16* d_A = NULL;
	__nv_bfloat16* d_B = NULL;
	float* d_C = NULL;
	
	double FLOPS = 0.0;
	double TIME = 0.0;
	
  cudaError_t cuerror;
	cublasStatus_t status;
  cublasHandle_t handle; 

  status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS initialization error\n");
	}

	status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH );  /* CUBLAS_DEFAULT_MATH */	
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS set MathMode error\n");
	}

	std::cout << std::endl;

	for (size_t i = 0; i <= r; i ++)
	{
		FLOPS = 2.0*((double)m)*((double)n)*((double)k);
		
		// init data on host
		h_A = new __nv_bfloat16[m*k];
		h_B = new __nv_bfloat16[k*n];
		h_C = new float[m*n];
		for (size_t j = 0; j < i*i; j++)
		{
			h_A[j] = __float2bfloat16((float)drand48());
			h_B[j] = __float2bfloat16((float)drand48());
			h_C[j] = 1.0;
		}
		
#ifndef MEASURE_KERNEL
		timer_start();
#endif
		
		cuerror = cudaMalloc((void**)&d_A, m*k*sizeof(__nv_bfloat16));
		if (cuerror != cudaSuccess) {
		printf ("!!!! device memory allocation error (A)\n");
		}

		cuerror = cudaMalloc((void**)&d_B, k*n*sizeof(__nv_bfloat16));
		if (cuerror != cudaSuccess) {
		printf ("!!!! device memory allocation error (B)\n");
		}

		cuerror = cudaMalloc((void**)&d_C, m*n*sizeof(float));
		if (cuerror != cudaSuccess) {
			printf ("!!!! device memory allocation error (C)\n");
		}

		status = cublasSetVector(m*k, sizeof(__nv_bfloat16), h_A, 1, d_A, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write A)\n");
		}

		status = cublasSetVector(k*n, sizeof(__nv_bfloat16), h_B, 1, d_B, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write B)\n");
		}

    status = cublasSetVector(m*n, sizeof(float), h_C, 1, d_C, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write C)\n");
		}
    __nv_bfloat16 alpha = __float2bfloat16(1.0);
		float beta  = 1.0;
#ifdef MEASURE_KERNEL
		timer_start();
#endif
    for ( int z = 0; z < s; ++z ) {
		  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, CUDA_R_16BF, m, d_B, CUDA_R_16BF, k, &beta, d_C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();
#ifdef MEASURE_KERNEL
		TIME = timer_stop();
#endif

		cuerror = cudaFree(d_A);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (A)\n");
		}

		cuerror = cudaFree(d_B);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (B)\n");
		}

		cuerror = cudaFree(d_C);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (C)\n");
		}
		
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
	
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! shutdown error\n");
	}
}

void test_bfloat_bfloat_general(size_t m, size_t n, size_t k, size_t s, size_t r)
{
	// host-side allocation
	__nv_bfloat16* h_A = NULL;
	__nv_bfloat16* h_B = NULL;
  __nv_bfloat16* h_C = NULL;

	// Allocate memory on GFX
	__nv_bfloat16* d_A = NULL;
	__nv_bfloat16* d_B = NULL;
	__nv_bfloat16* d_C = NULL;
	
	double FLOPS = 0.0;
	double TIME = 0.0;
	
  cudaError_t cuerror;
	cublasStatus_t status;
  cublasHandle_t handle; 

  status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS initialization error\n");
	}

	status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH );  /* CUBLAS_DEFAULT_MATH */	
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS set MathMode error\n");
	}

	std::cout << std::endl;

	for (size_t i = 0; i <= r; i ++)
	{
		FLOPS = 2.0*((double)m)*((double)n)*((double)k);
		
		// init data on host
		h_A = new __nv_bfloat16[m*k];
		h_B = new __nv_bfloat16[k*n];
		h_C = new __nv_bfloat16[m*n];
		for (size_t j = 0; j < i*i; j++)
		{
			h_A[j] = __float2bfloat16((float)drand48());
			h_B[j] = __float2bfloat16((float)drand48());
			h_C[j] = __float2bfloat16(1.0);
		}
		
#ifndef MEASURE_KERNEL
		timer_start();
#endif
		
		cuerror = cudaMalloc((void**)&d_A, m*k*sizeof(__nv_bfloat16));
		if (cuerror != cudaSuccess) {
		printf ("!!!! device memory allocation error (A)\n");
		}

		cuerror = cudaMalloc((void**)&d_B, k*n*sizeof(__nv_bfloat16));
		if (cuerror != cudaSuccess) {
		printf ("!!!! device memory allocation error (B)\n");
		}

		cuerror = cudaMalloc((void**)&d_C, m*n*sizeof(__nv_bfloat16));
		if (cuerror != cudaSuccess) {
			printf ("!!!! device memory allocation error (C)\n");
		}

		status = cublasSetVector(m*k, sizeof(__nv_bfloat16), h_A, 1, d_A, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write A)\n");
		}

		status = cublasSetVector(k*n, sizeof(__nv_bfloat16), h_B, 1, d_B, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write B)\n");
		}

    status = cublasSetVector(m*n, sizeof(__nv_bfloat16), h_C, 1, d_C, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write C)\n");
		}
    __nv_bfloat16 alpha = __float2bfloat16(1.0);
		__nv_bfloat16 beta  = __float2bfloat16(1.0);
#ifdef MEASURE_KERNEL
		timer_start();
#endif
    for ( int z = 0; z < s; ++z ) {
		  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, CUDA_R_16BF, m, d_B, CUDA_R_16BF, k, &beta, d_C, CUDA_R_16BF, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();
#ifdef MEASURE_KERNEL
		TIME = timer_stop();
#endif

		cuerror = cudaFree(d_A);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (A)\n");
		}

		cuerror = cudaFree(d_B);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (B)\n");
		}

		cuerror = cudaFree(d_C);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (C)\n");
		}
		
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
	
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! shutdown error\n");
	}
}

void test_float_general(size_t m, size_t n, size_t k, size_t s, size_t r, char use_tf32)
{
	// host-side allocation
	float* h_A = NULL;
	float* h_B = NULL;
	float* h_C = NULL;

	// Allocate memory on GFX
	float* d_A = NULL;
	float* d_B = NULL;
	float* d_C = NULL;
	
	double FLOPS = 0.0;
	double TIME = 0.0;
	
  cudaError_t cuerror;
	cublasStatus_t status;
  cublasHandle_t handle; 

  status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS initialization error\n");
	}

  if ( use_tf32 != 0 ) {
    status = cublasSetMathMode( handle, CUBLAS_TF32_TENSOR_OP_MATH );
  } else {
    status = cublasSetMathMode( handle, CUBLAS_DEFAULT_MATH );  /* CUBLAS_DEFAULT_MATH */	
  }
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS set MathMode error\n");
	}

	std::cout << std::endl;

	for (size_t i = 0; i <= r; i ++)
	{
		FLOPS = 2.0*((double)m)*((double)n)*((double)k);
		
		// init data on host
		h_A = new float[m*k];
		h_B = new float[k*n];
		h_C = new float[m*n];
		
		for (size_t j = 0; j < i*i; j++)
		{
			h_A[j] = (float)drand48();
			h_B[j] = (float)drand48();
			h_C[j] = 1.0;
		}
		
#ifndef MEASURE_KERNEL
		timer_start();
#endif
		
		cuerror = cudaMalloc((void**)&d_A, m*k*sizeof(float));
		if (cuerror != cudaSuccess) {
			printf ("!!!! device memory allocation error (A)\n");
		}

		cuerror = cudaMalloc((void**)&d_B, k*n*sizeof(float));
		if (cuerror != cudaSuccess) {
			printf ("!!!! device memory allocation error (B)\n");
		}

		cuerror = cudaMalloc((void**)&d_C, m*n*sizeof(float));
		if (cuerror != cudaSuccess) {
			printf ("!!!! device memory allocation error (C)\n");
		}

		status = cublasSetVector(m*k, sizeof(float), h_A, 1, d_A, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write A)\n");
		}

		status = cublasSetVector(k*n, sizeof(float), h_B, 1, d_B, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write B)\n");
		}

		status = cublasSetVector(m*n, sizeof(float), h_C, 1, d_C, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write C)\n");
		}

                float alpha = 1.0;
		float beta  = 0.0;
#ifdef MEASURE_KERNEL
		timer_start();
#endif
    for ( int z = 0; z < s; ++z ) {
#if 1
		  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
#else
      if ( use_tf32 != 0 ) {
		    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, CUDA_R_32F, m, d_B, CUDA_R_32F, k, &beta, d_C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      } else {
		    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
      }
#endif
    }
    cudaDeviceSynchronize();
#ifdef MEASURE_KERNEL
		TIME = timer_stop();
#endif

		cuerror = cudaFree(d_A);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (A)\n");
		}

		cuerror = cudaFree(d_B);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (B)\n");
		}

		cuerror = cudaFree(d_C);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (C)\n");
		}
		
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
	
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! shutdown error\n");
	}
}


void test_double_general(size_t m, size_t n, size_t k, size_t s, size_t r)
{
	// host-side allocation
	double* h_A = NULL;
	double* h_B = NULL;
	double* h_C = NULL;

	// Allocate memory on GFX
	double* d_A = NULL;
	double* d_B = NULL;
	double* d_C = NULL;
	
	double FLOPS = 0.0;
	double TIME = 0.0;
	
  cudaError_t cuerror;
	cublasStatus_t status;
  cublasHandle_t handle; 

  status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS initialization error\n");
	}

	status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH );  /* CUBLAS_DEFAULT_MATH */	
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS set MathMode error\n");
	}

	std::cout << std::endl;

	for (size_t i = 0; i <= r; i ++)
	{
		FLOPS = 2.0*((double)m)*((double)n)*((double)k);
		
		// init data on host
		h_A = new double[m*k];
		h_B = new double[k*n];
		h_C = new double[m*n];
		
		for (size_t j = 0; j < i*i; j++)
		{
			h_A[j] = drand48();
			h_B[j] = drand48();
			h_C[j] = 1.0;
		}
		
#ifndef MEASURE_KERNEL
		timer_start();
#endif
		
		cuerror = cudaMalloc((void**)&d_A, m*k*sizeof(double));
		if (cuerror != cudaSuccess) {
			printf ("!!!! device memory allocation error (A)\n");
		}

		cuerror = cudaMalloc((void**)&d_B, k*n*sizeof(double));
		if (cuerror != cudaSuccess) {
			printf ("!!!! device memory allocation error (B)\n");
		}

		cuerror = cudaMalloc((void**)&d_C, m*n*sizeof(double));
		if (cuerror != cudaSuccess) {
			printf ("!!!! device memory allocation error (C)\n");
		}

		status = cublasSetVector(m*k, sizeof(double), h_A, 1, d_A, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write A)\n");
		}

		status = cublasSetVector(k*n, sizeof(double), h_B, 1, d_B, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write B)\n");
		}

		status = cublasSetVector(m*n, sizeof(double), h_C, 1, d_C, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write C)\n");
		}

		double alpha = 1.0;
		double beta  = 0.0;
#ifdef MEASURE_KERNEL
		timer_start();
#endif
    for ( int z = 0; z < s; ++z ) {
		  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    }
    cudaDeviceSynchronize();
#ifdef MEASURE_KERNEL
		TIME = timer_stop();
#endif

		cuerror = cudaFree(d_A);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (A)\n");
		}

		cuerror = cudaFree(d_B);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (B)\n");
		}

		cuerror = cudaFree(d_C);
		if (cuerror != cudaSuccess) {
			printf ("!!!! memory free error (C)\n");
		}
		
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
	
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! shutdown error\n");
	}
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
	
  cudaSetDevice(dev);
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
