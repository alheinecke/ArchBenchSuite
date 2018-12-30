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
#include <cstdio>
#include <pmmintrin.h>
#include <sys/time.h>
#include <cublas.h>

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

void test_float(size_t start, size_t end, size_t inc)
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
	
	cublasStatus status;

	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS initialization error\n");
	}
	
	std::cout << std::endl;

	for (size_t i = start; i <= end; i += inc)
	{
		FLOPS = 2.0*((double)i)*((double)i)*((double)i);
		
		// init data on host
		h_A = new float[i*i];
		h_B = new float[i*i];
		h_C = new float[i*i];
		
		for (size_t j = 0; j < i*i; j++)
		{
			h_A[j] = 1.0;
			h_B[j] = 1.0;
			h_C[j] = 1.0;
		}
		
#ifndef MEASURE_KERNEL
		timer_start();
#endif
		
		status = cublasAlloc(i*i, sizeof(float), (void**)&d_A);
		if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! device memory allocation error (A)\n");
		}

		status = cublasAlloc(i*i, sizeof(float), (void**)&d_B);
		if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! device memory allocation error (B)\n");
		}

		status = cublasAlloc(i*i, sizeof(float), (void**)&d_C);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device memory allocation error (C)\n");
		}

		status = cublasSetVector(i*i, sizeof(float), h_A, 1, d_A, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write A)\n");
		}

		status = cublasSetVector(i*i, sizeof(float), h_B, 1, d_B, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write B)\n");
		}

		status = cublasSetVector(i*i, sizeof(float), h_C, 1, d_C, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write C)\n");
		}

#ifdef MEASURE_KERNEL
		timer_start();
#endif
                for ( int z = 0; z < 100; ++z ) {
		  cublasSgemm('n', 'n', i, i, i, 1.0, d_A, i, d_B, i, 0.0, d_C, i);
                }
                cudaDeviceSynchronize();
#ifdef MEASURE_KERNEL
		TIME = timer_stop();
#endif

		status = cublasFree(d_A);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! memory free error (A)\n");
		}

		status = cublasFree(d_B);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! memory free error (B)\n");
		}

		status = cublasFree(d_C);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! memory free error (C)\n");
		}
		
#ifndef MEASURE_KERNEL
		TIME = timer_stop();
#endif		
	
		delete[] h_A;
		delete[] h_B;
		delete[] h_C;
		
		// Print results
		std::cout << i << ";" << TIME/100.0 << ";" << (FLOPS/(1e9))/(TIME/100.0) << std::endl;
	}
	
	std::cout << std::endl;
	
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! shutdown error\n");
	}
}

void test_float_general(size_t m, size_t n, size_t k, size_t s, size_t r)
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
	
	cublasStatus status;

	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS initialization error\n");
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
			h_A[j] = 1.0;
			h_B[j] = 1.0;
			h_C[j] = 1.0;
		}
		
#ifndef MEASURE_KERNEL
		timer_start();
#endif
		
		status = cublasAlloc(m*k, sizeof(float), (void**)&d_A);
		if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! device memory allocation error (A)\n");
		}

		status = cublasAlloc(k*n, sizeof(float), (void**)&d_B);
		if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! device memory allocation error (B)\n");
		}

		status = cublasAlloc(m*n, sizeof(float), (void**)&d_C);
		if (status != CUBLAS_STATUS_SUCCESS) {
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

#ifdef MEASURE_KERNEL
		timer_start();
#endif
                for ( int z = 0; z < s; ++z ) {
		  cublasSgemm('n', 'n', m, n, k, 1.0, d_A, m, d_B, k, 0.0, d_C, m);
                }
                cudaDeviceSynchronize();
#ifdef MEASURE_KERNEL
		TIME = timer_stop();
#endif

		status = cublasFree(d_A);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! memory free error (A)\n");
		}

		status = cublasFree(d_B);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! memory free error (B)\n");
		}

		status = cublasFree(d_C);
		if (status != CUBLAS_STATUS_SUCCESS) {
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
	
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! shutdown error\n");
	}
}


void test_double(size_t start, size_t end, size_t inc)
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
	
	cublasStatus status;

	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS initialization error\n");
	}
	
	std::cout << std::endl;

	for (size_t i = start; i <= end; i += inc)
	{
		FLOPS = 2.0*((double)i)*((double)i)*((double)i);
		
		// init data on host
		h_A = new double[i*i];
		h_B = new double[i*i];
		h_C = new double[i*i];
		
		for (size_t j = 0; j < i*i; j++)
		{
			h_A[j] = 1.0;
			h_B[j] = 1.0;
			h_C[j] = 1.0;
		}
		
#ifndef MEASURE_KERNEL
		timer_start();
#endif
		
		status = cublasAlloc(i*i, sizeof(double), (void**)&d_A);
		if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! device memory allocation error (A)\n");
		}

		status = cublasAlloc(i*i, sizeof(double), (void**)&d_B);
		if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! device memory allocation error (B)\n");
		}

		status = cublasAlloc(i*i, sizeof(double), (void**)&d_C);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device memory allocation error (C)\n");
		}

		status = cublasSetVector(i*i, sizeof(double), h_A, 1, d_A, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write A)\n");
		}

		status = cublasSetVector(i*i, sizeof(double), h_B, 1, d_B, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write B)\n");
		}

		status = cublasSetVector(i*i, sizeof(double), h_C, 1, d_C, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! device access error (write C)\n");
		}

#ifdef MEASURE_KERNEL
		timer_start();
#endif
                for ( int z = 0; z < 100; ++z ) {
	      	  cublasDgemm('n', 'n', i, i, i, 1.0, d_A, i, d_B, i, 0.0, d_C, i);
                }
                cudaDeviceSynchronize();
#ifdef MEASURE_KERNEL
		TIME = timer_stop();
#endif

		status = cublasFree(d_A);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! memory free error (A)\n");
		}

		status = cublasFree(d_B);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! memory free error (B)\n");
		}

		status = cublasFree(d_C);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! memory free error (C)\n");
		}
		
#ifndef MEASURE_KERNEL
		TIME = timer_stop();
#endif		
	
		delete[] h_A;
		delete[] h_B;
		delete[] h_C;
		
		// Print results
		std::cout << i << ";" << TIME/100.0 << ";" << (FLOPS/(1e9))/(TIME/100.0) << std::endl;
	}
	
	std::cout << std::endl;
	
	status = cublasShutdown();
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
	
	cublasStatus status;

	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! CUBLAS initialization error\n");
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
			h_A[j] = 1.0;
			h_B[j] = 1.0;
			h_C[j] = 1.0;
		}
		
#ifndef MEASURE_KERNEL
		timer_start();
#endif
		
		status = cublasAlloc(m*k, sizeof(double), (void**)&d_A);
		if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! device memory allocation error (A)\n");
		}

		status = cublasAlloc(k*n, sizeof(double), (void**)&d_B);
		if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! device memory allocation error (B)\n");
		}

		status = cublasAlloc(m*n, sizeof(double), (void**)&d_C);
		if (status != CUBLAS_STATUS_SUCCESS) {
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

#ifdef MEASURE_KERNEL
		timer_start();
#endif
                for ( int z = 0; z < s; ++z ) {
		  cublasDgemm('n', 'n', m, n, k, 1.0, d_A, m, d_B, k, 0.0, d_C, m);
                }
                cudaDeviceSynchronize();
#ifdef MEASURE_KERNEL
		TIME = timer_stop();
#endif

		status = cublasFree(d_A);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! memory free error (A)\n");
		}

		status = cublasFree(d_B);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf ("!!!! memory free error (B)\n");
		}

		status = cublasFree(d_C);
		if (status != CUBLAS_STATUS_SUCCESS) {
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
	
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf ("!!!! shutdown error\n");
	}
}

void write_help()
{
	std::cout << std::endl << "Wrong parameters! Please use:" << std::endl;
	std::cout << "  device number" << std::endl;
	std::cout << "	0: float; 1: double" << std::endl;
	std::cout << "	start" << std::endl;
	std::cout << "	stop" << std::endl;
	std::cout << "	inc" << std::endl << std::endl;
	std::cout << "Example: 0 0 32 4096 32" << std::endl << std::endl;
}

void write_help_general()
{
	std::cout << std::endl << "Wrong parameters! Please use:" << std::endl;
	std::cout << "  device number" << std::endl;
	std::cout << "	0: float; 1: double" << std::endl;
	std::cout << "	m" << std::endl;
	std::cout << "	n" << std::endl;
	std::cout << "	k" << std::endl;
        std::cout << "  s" << std::endl;
        std::cout << "  r" << std::endl << std::endl;
	std::cout << "Example: 0 0 4096 448 2048" << std::endl << std::endl;
}


#if 0
int main(int argc, char* argv[])
{
	if (argc != 6 )
	{
		write_help();
		return 0;
	}

        size_t dev = atoi(argv[1]);	
	size_t mode = atoi(argv[2]);
	size_t start = atoi(argv[3]);
	size_t stop = atoi(argv[4]);
	size_t inc = atoi(argv[5]);
	
        cudaSetDevice(dev);

	if (mode == 0)
	{
		test_float(start, stop, inc);
	}
	else if (mode == 1)
	{
		test_double(start, stop, inc);
	}
	else
	{
		write_help();
		return 1;
	}

	return 0;
}
#else
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

	if (mode == 0)
	{
		test_float_general(m, n, k, s, r);
	}
	else if (mode == 1)
	{
		test_double_general(m, n, k, s, r);
	}
	else
	{
		write_help_general();
		return 1;
	}

	return 0;
}
#endif
