/* This is a modified version of stream */
/*-----------------------------------------------------------------------*/
/* Program: STREAM                                                       */
/* Revision: $Id: stream.c,v 5.10 2013/01/17 16:01:06 mccalpin Exp mccalpin $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2013: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*           "tuned STREAM benchmark results"                            */
/*           "based on a variant of the STREAM benchmark code"           */
/*         Other comparable, clear, and reasonable labelling is          */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/
# include <stdio.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
# include <unistd.h>

#ifdef BENCH_RV64
# include <riscv_vector.h>
# include <omp.h>
#endif

/*-----------------------------------------------------------------------
 * INSTRUCTIONS:
 *
 *	1) STREAM requires different amounts of memory to run on different
 *           systems, depending on both the system cache size(s) and the
 *           granularity of the system timer.
 *     You should adjust the value of 'STREAM_ARRAY_SIZE' (below)
 *           to meet *both* of the following criteria:
 *       (a) Each array must be at least 4 times the size of the
 *           available cache memory. I don't worry about the difference
 *           between 10^6 and 2^20, so in practice the minimum array size
 *           is about 3.8 times the cache size.
 *           Example 1: One Xeon E3 with 8 MB L3 cache
 *               STREAM_ARRAY_SIZE should be >= 4 million, giving
 *               an array size of 30.5 MB and a total memory requirement
 *               of 91.5 MB.  
 *           Example 2: Two Xeon E5's with 20 MB L3 cache each (using OpenMP)
 *               STREAM_ARRAY_SIZE should be >= 20 million, giving
 *               an array size of 153 MB and a total memory requirement
 *               of 458 MB.  
 *       (b) The size should be large enough so that the 'timing calibration'
 *           output by the program is at least 20 clock-ticks.  
 *           Example: most versions of Windows have a 10 millisecond timer
 *               granularity.  20 "ticks" at 10 ms/tic is 200 milliseconds.
 *               If the chip is capable of 10 GB/s, it moves 2 GB in 200 msec.
 *               This means the each array must be at least 1 GB, or 128M elements.
 *
 *      Version 5.10 increases the default array size from 2 million
 *          elements to 10 million elements in response to the increasing
 *          size of L3 caches.  The new default size is large enough for caches
 *          up to 20 MB. 
 *      Version 5.10 changes the loop index variables from "register int"
 *          to "ssize_t", which allows array indices >2^32 (4 billion)
 *          on properly configured 64-bit systems.  Additional compiler options
 *          (such as "-mcmodel=medium") may be required for large memory runs.
 *
 *      Array size can be set at compile time without modifying the source
 *          code for the (many) compilers that support preprocessor definitions
 *          on the compile line.  E.g.,
 *                gcc -O -DSTREAM_ARRAY_SIZE=100000000 stream.c -o stream.100M
 *          will override the default size of 10M with a new size of 100M elements
 *          per array.
 */
#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	10000000
#endif

/*  2) STREAM runs each kernel "NTIMES" times and reports the *best* result
 *         for any iteration after the first, therefore the minimum value
 *         for NTIMES is 2.
 *      There are no rules on maximum allowable values for NTIMES, but
 *         values larger than the default are unlikely to noticeably
 *         increase the reported performance.
 *      NTIMES can also be set on the compile line without changing the source
 *         code using, for example, "-DNTIMES=7".
 */
#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES	10
#endif
#endif
#ifndef NTIMES
#   define NTIMES	10
#endif

/*  Users are allowed to modify the "OFFSET" variable, which *may* change the
 *         relative alignment of the arrays (though compilers may change the 
 *         effective offset by making the arrays non-contiguous on some systems). 
 *      Use of non-zero values for OFFSET can be especially helpful if the
 *         STREAM_ARRAY_SIZE is set to a value close to a large power of 2.
 *      OFFSET can also be set on the compile line without changing the source
 *         code using, for example, "-DOFFSET=56".
 */
#ifndef OFFSET
#   define OFFSET	0
#endif

/*
 *	3) Compile the code with optimization.  Many compilers generate
 *       unreasonably bad code before the optimizer tightens things up.  
 *     If the results are unreasonably good, on the other hand, the
 *       optimizer might be too smart for me!
 *
 *     For a simple single-core version, try compiling with:
 *            cc -O stream.c -o stream
 *     This is known to work on many, many systems....
 *
 *     To use multiple cores, you need to tell the compiler to obey the OpenMP
 *       directives in the code.  This varies by compiler, but a common example is
 *            gcc -O -fopenmp stream.c -o stream_omp
 *       The environment variable OMP_NUM_THREADS allows runtime control of the 
 *         number of threads/cores used when the resulting "stream_omp" program
 *         is executed.
 *
 *     To run with single-precision variables and arithmetic, simply add
 *         -DSTREAM_TYPE=float
 *     to the compile line.
 *     Note that this changes the minimum array sizes required --- see (1) above.
 *
 *     The preprocessor directive "TUNED" does not do much -- it simply causes the 
 *       code to call separate functions to execute each kernel.  Trivial versions
 *       of these functions are provided, but they are *not* tuned -- they just 
 *       provide predefined interfaces to be replaced with tuned code.
 *
 *
 *	4) Optional: Mail the results to mccalpin@cs.virginia.edu
 *	   Be sure to include info that will help me understand:
 *		a) the computer hardware configuration (e.g., processor model, memory type)
 *		b) the compiler name/version and compilation flags
 *      c) any run-time information (such as OMP_NUM_THREADS)
 *		d) all of the output from the test case.
 *
 * Thanks!
 *
 *-----------------------------------------------------------------------*/

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

#if 0
#define USESTACK
#endif

#if 1
#define USE_AVX512_PREFETCH
#endif

#ifdef USESTACK
static   STREAM_TYPE	a[STREAM_ARRAY_SIZE+OFFSET],
			b[STREAM_ARRAY_SIZE+OFFSET],
			c[STREAM_ARRAY_SIZE+OFFSET];
#else
#include <stdlib.h>
STREAM_TYPE* a;
STREAM_TYPE* b;
STREAM_TYPE* c;
#endif

static double	avgtime[4] = {0}, maxtime[4] = {0},
		mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char	*label[4] = {"Copy:      ", "Scale:     ",
    "Add:       ", "Triad:     "};

static double	bytes[4] = {
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE
    };

extern double mysecond();
extern void checkSTREAMresults();

#ifdef __SSE3__
#include <immintrin.h>
#include <omp.h>
#define TUNED
#endif
#ifdef BENCH_POWER8
#include <altivec.h>
#define TUNED
#endif
#ifdef BENCH_ARMV8
#define TUNED
#endif
#ifdef BENCH_RV64
#define TUNED
#endif


#ifdef USE_CUDA_HMM
#include <cuda.h>

#define CUDA_THREAD_PER_BLOCK 256
#define TUNED
__global__ void tuned_STREAM_Copy_cuda(double* d_a, double* d_c);
__global__ void tuned_STREAM_Scale_cuda(double* d_c, double* d_b);
__global__ void tuned_STREAM_Add_cuda(double* d_a, double* d_b, double* d_c);
__global__ void tuned_STREAM_Triad_cuda(double* d_a, double* d_b, double* d_c);

__global__ void tuned_STREAM_Copy_cuda(double* d_a, double* d_c) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < STREAM_ARRAY_SIZE) d_c[id] = d_a[id];
}

__global__ void tuned_STREAM_Scale_cuda(double* d_c, double* d_b) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < STREAM_ARRAY_SIZE) d_b[id] = 3.0 * d_c[id];
}

__global__ void tuned_STREAM_Add_cuda(double* d_a, double* d_b, double* d_c) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < STREAM_ARRAY_SIZE) d_c[id] = d_a[id] + d_b[id];
}

__global__ void tuned_STREAM_Triad_cuda(double* d_a, double* d_b, double* d_c) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < STREAM_ARRAY_SIZE) d_a[id] = d_b[id] + 3.0 * d_c[id];
}
#endif

#ifdef USE_SYCL_USM
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/memory.hpp>

sycl::queue sycl_q; // persistent SYCL queue

#define TUNED
constexpr int VL = 8;  // Vector length (aligned for 64 byte cache line) 
static_assert(STREAM_ARRAY_SIZE % (2*VL) == 0, "STREAM_ARRAY_SIZE must be a multiple of 2*VL");

void tuned_STREAM_Copy_sycl(sycl::queue& q, double* d_a, double* d_c) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for<class StreamCopyESIMD>(
            sycl::nd_range<1>(
                sycl::range<1>{STREAM_ARRAY_SIZE / (2 * VL)},
                sycl::range<1>{64}
            ),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                int i = item.get_global_id(0) * 2 * VL;

                sycl::ext::intel::esimd::simd<double, VL> va1(d_a + i);
                sycl::ext::intel::esimd::simd<double, VL> va2(d_a + i + VL);

                va1.copy_to(d_c + i);
                va2.copy_to(d_c + i + VL);
            }
        );
    });
}

void tuned_STREAM_Scale_sycl(sycl::queue& q, double* d_c, double* d_b) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for<class StreamScaleESIMD>(
            sycl::nd_range<1>(
                sycl::range<1>{STREAM_ARRAY_SIZE / (2 * VL)},
                sycl::range<1>{64} 
            ),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                int i = item.get_global_id(0) * 2 * VL;

                sycl::ext::intel::esimd::simd<double, VL> vc1(d_c + i);
                sycl::ext::intel::esimd::simd<double, VL> vc2(d_c + i + VL);

                (vc1 * 3.0).copy_to(d_b + i);
                (vc2 * 3.0).copy_to(d_b + i + VL);
            }
        );
    });
}

void tuned_STREAM_Add_sycl(sycl::queue& q, double* d_a, double* d_b, double* d_c) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for<class StreamAddESIMD>(
            sycl::nd_range<1>(
                sycl::range<1>{STREAM_ARRAY_SIZE / (2 * VL)},
                sycl::range<1>{64}
            ),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                int i = item.get_global_id(0) * 2 * VL;

                sycl::ext::intel::esimd::simd<double, VL> va1(d_a + i);
                sycl::ext::intel::esimd::simd<double, VL> vb1(d_b + i);
                (va1 + vb1).copy_to(d_c + i);

                sycl::ext::intel::esimd::simd<double, VL> va2(d_a + i + VL);
                sycl::ext::intel::esimd::simd<double, VL> vb2(d_b + i + VL);
                (va2 + vb2).copy_to(d_c + i + VL);
            }
        );
    });
}


void tuned_STREAM_Triad_sycl(sycl::queue& q, double* d_a, double* d_b, double* d_c) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for<class StreamTriadESIMD>(
            sycl::nd_range<1>(
                sycl::range<1>{STREAM_ARRAY_SIZE / (2 * VL)},
                sycl::range<1>{64}
            ),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                int i = item.get_global_id(0) * 2 * VL;

                sycl::ext::intel::esimd::simd<double, VL> vb1(d_b + i);
                sycl::ext::intel::esimd::simd<double, VL> vc1(d_c + i);
                (vb1 + vc1 * 3.0).copy_to(d_a + i);

                sycl::ext::intel::esimd::simd<double, VL> vb2(d_b + i + VL);
                sycl::ext::intel::esimd::simd<double, VL> vc2(d_c + i + VL);
                (vb2 + vc2 * 3.0).copy_to(d_a + i + VL);
            }
        );
    });
}

#endif


#ifdef TUNED
extern void tuned_STREAM_Copy();
extern void tuned_STREAM_Scale(STREAM_TYPE scalar);
extern void tuned_STREAM_Add();
extern void tuned_STREAM_Triad(STREAM_TYPE scalar);
#endif
#ifdef _OPENMP
extern int omp_get_num_threads();
#endif
int
main()
    {
    int			quantum, checktick();
    int			BytesPerWord;
    int			k;
    ssize_t		j;
    STREAM_TYPE		scalar;
    double		t, times[4][NTIMES];

    /* --- SETUP --- determine precision and check timing --- */

#ifndef USESTACK
    posix_memalign((void**)&a, 2097152, ((size_t)STREAM_ARRAY_SIZE)*sizeof(double));
    posix_memalign((void**)&b, 2097152, ((size_t)STREAM_ARRAY_SIZE)*sizeof(double));
    posix_memalign((void**)&c, 2097152, ((size_t)STREAM_ARRAY_SIZE)*sizeof(double));
#endif

    printf(HLINE);
    printf("STREAM version $Revision: 5.10 $\n");
    printf(HLINE);
    BytesPerWord = sizeof(STREAM_TYPE);
    printf("This system uses %d bytes per array element.\n",
	BytesPerWord);

    printf(HLINE);
#ifdef N
    printf("*****  WARNING: ******\n");
    printf("      It appears that you set the preprocessor variable N when compiling this code.\n");
    printf("      This version of the code uses the preprocesor variable STREAM_ARRAY_SIZE to control the array size\n");
    printf("      Reverting to default value of STREAM_ARRAY_SIZE=%llu\n",(unsigned long long) STREAM_ARRAY_SIZE);
    printf("*****  WARNING: ******\n");
#endif

    printf("Array size = %llu (elements), Offset = %d (elements)\n" , (unsigned long long) STREAM_ARRAY_SIZE, OFFSET);
    printf("Memory per array = %.1f MiB (= %.1f GiB).\n", 
	BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0),
	BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0/1024.0));
    printf("Total memory required = %.1f MiB (= %.1f GiB).\n",
	(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.),
	(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024./1024.));
    printf("Each kernel will be executed %d times.\n", NTIMES);
    printf(" The *best* time for each kernel (excluding the first iteration)\n"); 
    printf(" will be used to compute the reported bandwidth.\n");

#ifdef _OPENMP
    printf(HLINE);
#pragma omp parallel 
    {
#pragma omp master
	{
	    k = omp_get_num_threads();
	    printf ("Number of Threads requested = %i\n",k);
        }
    }
#else
    k = 1;
#endif

#ifdef _OPENMP
	k = 0;
#pragma omp parallel
#pragma omp atomic 
		k++;
    printf ("Number of Threads counted = %i\n",k);
#endif

    /* check for array size for simple load balancing */
    if ( STREAM_ARRAY_SIZE % (k*512) != 0 ) {
      printf("Please define STREAM_ARRAY_SIZE such that it is divisable by number_of_threads*512!\n Exiting...\n");
      return -1;
    } 

    /* Get initial value for system clock. */
#pragma omp parallel for
    for (j=0; j<STREAM_ARRAY_SIZE; j++) {
	    a[j] = 1.0;
	    b[j] = 2.0;
	    c[j] = 0.0;
	}

    printf(HLINE);

    if  ( (quantum = checktick()) >= 1) 
	printf("Your clock granularity/precision appears to be "
	    "%d microseconds.\n", quantum);
    else {
	printf("Your clock granularity appears to be "
	    "less than one microsecond.\n");
	quantum = 1;
    }

    t = mysecond();
#pragma omp parallel for
    for (j = 0; j < STREAM_ARRAY_SIZE; j++)
		a[j] = 2.0E0 * a[j];
    t = 1.0E6 * (mysecond() - t);

    printf("Each test below will take on the order"
	" of %d microseconds.\n", (int) t  );
    printf("   (= %d clock ticks)\n", (int) (t/quantum) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);
    
    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

    scalar = 3.0;
    for (k=0; k<NTIMES; k++)
	{
	times[0][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Copy();
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    c[j] = a[j];
#endif
	times[0][k] = mysecond() - times[0][k];
	
	times[1][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Scale(scalar);
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    b[j] = scalar*c[j];
#endif
	times[1][k] = mysecond() - times[1][k];
	
	times[2][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Add();
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    c[j] = a[j]+b[j];
#endif
	times[2][k] = mysecond() - times[2][k];
	
	times[3][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Triad(scalar);
#else
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++) {
	    a[j] = b[j]+scalar*c[j];
        }
#endif
	times[3][k] = mysecond() - times[3][k];
	}

    /*	--- SUMMARY --- */

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
	for (j=0; j<4; j++)
	    {
	    avgtime[j] = avgtime[j] + times[j][k];
	    mintime[j] = MIN(mintime[j], times[j][k]);
	    maxtime[j] = MAX(maxtime[j], times[j][k]);
	    }
	}
    
    printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
    char hostname[255];
    gethostname(hostname, 255);
    for (j=0; j<4; j++) {
		avgtime[j] = avgtime[j]/(double)(NTIMES-1);

		printf("%s %s%12.1f  %11.6f  %11.6f  %11.6f\n", hostname, label[j],
	       1.0E-06 * bytes[j]/mintime[j],
	       avgtime[j],
	       mintime[j],
	       maxtime[j]);
    }
    printf(HLINE);

    /* --- Check Results --- */
    checkSTREAMresults();
    printf(HLINE);

    return 0;
}

# define	M	20

int
checktick()
    {
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++) {
	t1 = mysecond();
	while( ((t2=mysecond()) - t1) < 1.0E-6 )
	    ;
	timesfound[i] = t1 = t2;
	}

/*
 * Determine the minimum difference between these M values.
 * This result will be our estimate (in microseconds) for the
 * clock granularity.
 */

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
	Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
	minDelta = MIN(minDelta, MAX(Delta,0));
	}

   return(minDelta);
    }



/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <sys/time.h>

double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif
void checkSTREAMresults ()
{
	STREAM_TYPE aj,bj,cj,scalar;
	STREAM_TYPE aSumErr,bSumErr,cSumErr;
	STREAM_TYPE aAvgErr,bAvgErr,cAvgErr;
	double epsilon;
	ssize_t	j;
	int	k,ierr,err;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;
    /* now execute timing loop */
	scalar = 3.0;
	for (k=0; k<NTIMES; k++)
        {
            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;
        }

    /* accumulate deltas between observed and expected results */
	aSumErr = 0.0;
	bSumErr = 0.0;
	cSumErr = 0.0;
	for (j=0; j<STREAM_ARRAY_SIZE; j++) {
		aSumErr += abs(a[j] - aj);
		bSumErr += abs(b[j] - bj);
		cSumErr += abs(c[j] - cj);
		// if (j == 417) printf("Index 417: c[j]: %f, cj: %f\n",c[j],cj);	// MCCALPIN
	}
	aAvgErr = aSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	bAvgErr = bSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	cAvgErr = cSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;

	if (sizeof(STREAM_TYPE) == 4) {
		epsilon = 1.e-6;
	}
	else if (sizeof(STREAM_TYPE) == 8) {
		epsilon = 1.e-13;
	}
	else {
		printf("WEIRD: sizeof(STREAM_TYPE) = %lu\n",sizeof(STREAM_TYPE));
		epsilon = 1.e-6;
	}

	err = 0;
	if (abs(aAvgErr/aj) > epsilon) {
		err++;
		printf ("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",aj,aAvgErr,abs(aAvgErr)/aj);
		ierr = 0;
		for (j=0; j<STREAM_ARRAY_SIZE; j++) {
			if (abs(a[j]/aj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array a: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,aj,a[j],abs((aj-a[j])/aAvgErr));
				}
#endif
			}
		}
		printf("     For array a[], %d errors were found.\n",ierr);
	}
	if (abs(bAvgErr/bj) > epsilon) {
		err++;
		printf ("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",bj,bAvgErr,abs(bAvgErr)/bj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (j=0; j<STREAM_ARRAY_SIZE; j++) {
			if (abs(b[j]/bj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array b: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,bj,b[j],abs((bj-b[j])/bAvgErr));
				}
#endif
			}
		}
		printf("     For array b[], %d errors were found.\n",ierr);
	}
	if (abs(cAvgErr/cj) > epsilon) {
		err++;
		printf ("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",cj,cAvgErr,abs(cAvgErr)/cj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (j=0; j<STREAM_ARRAY_SIZE; j++) {
			if (abs(c[j]/cj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array c: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,cj,c[j],abs((cj-c[j])/cAvgErr));
				}
#endif
			}
		}
		printf("     For array c[], %d errors were found.\n",ierr);
	}
	if (err == 0) {
		printf ("Solution Validates: avg error less than %e on all three arrays\n",epsilon);
	}
#ifdef VERBOSE
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",a[1],b[1],c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
#endif
}

#ifdef TUNED
/* stubs for "tuned" versions of the kernels */
void tuned_STREAM_Copy()
{
#ifdef USE_CUDA_HMM
	int thr_per_blk = CUDA_THREAD_PER_BLOCK;
	int blk_in_grid = ceil( float(STREAM_ARRAY_SIZE) / thr_per_blk );

  tuned_STREAM_Copy_cuda<<< blk_in_grid, thr_per_blk >>>(a, c);
  cudaDeviceSynchronize();
#elif defined(USE_SYCL_USM)
#if 0
    /* print GPU Name and USM */
    std::cout << "Using SYCL queue with device: "
              << q.get_device().get_info<sycl::info::device::name>() << std::endl;
 
    /* write with sycl::aspect::usm_system_allocations */
    std::cout << "Using SYCL queue with USM system allocations: "
              << q.get_device().has(sycl::aspect::usm_system_allocations) << "\n";
#endif
    tuned_STREAM_Copy_sycl(sycl_q, a, c);
    sycl_q.wait();
#else
        #pragma omp parallel
        {
          ssize_t j;
#ifdef _OPENMP
          ssize_t chunk = STREAM_ARRAY_SIZE/omp_get_num_threads();
          ssize_t start = chunk*omp_get_thread_num();
#else
          ssize_t chunk = STREAM_ARRAY_SIZE;
          ssize_t start = 0;
#endif          

#ifdef BENCH_AVX512
	  for (j=start; j< start+chunk; j+=8) {
#ifdef USE_AVX512_PREFETCH
            _mm_prefetch( (void*) &a[j+128], _MM_HINT_T2 );
            _mm_prefetch( (void*) &a[j+16], _MM_HINT_T1 );
#endif
            _mm512_stream_pd(&c[j], _mm512_load_pd(&a[j]));
    }
#endif
#ifdef BENCH_AVX
	  for (j=start; j< start+chunk; j+=4)
            _mm256_stream_pd(&c[j], _mm256_load_pd(&a[j]));
#endif
#ifdef BENCH_SSE
	  for (j=start; j< start+chunk; j+=2)
            _mm_stream_pd(&c[j], _mm_load_pd(&a[j]));
#endif
#ifdef BENCH_POWER8
	  for (j=start; j< start+chunk; j+=2)
            vec_vsx_st(vec_vsx_ld(0, &a[j]), 0, &c[j]);
#endif
#ifdef BENCH_RV64
    size_t gvl = __riscv_vsetvl_e64m1(16);
	  for (j=start; j< start+chunk; j+=2){
            __riscv_vse64_v_f64m1(&c[j], __riscv_vle64_v_f64m1((&a[j]), gvl), gvl);
    }
#endif
#ifdef BENCH_ARMV8
        __asm__ __volatile__("mov x0, %0\n\t"  // a
                             "mov x1, %1\n\t"  // c
                             "mov x4, %2\n\t"  // chunk
                             "1:\n\t"                            
                             "ldr  d0, [x0]\n\t"
                             "ldr  d1, [x0,8]\n\t"
                             "stnp d0, d1, [x1]\n\t" 
                             "ldr  d2, [x0,16]\n\t"
                             "ldr  d3, [x0,24]\n\t"
                             "stnp d2, d3, [x1,16]\n\t" 
                             "add x0, x0, #32\n\t"
                             "add x1, x1, #32\n\t"
                             "sub x4, x4, #4\n\t"
                             "cbnz x4, 1b\n\t"
                        : : "r" (&a[start]), "r" (&c[start]), "r" (chunk) : "x0","x1","x4","d0","d1","d2","d3"); 
#endif
#ifndef BENCH_AVX512
#ifndef BENCH_AVX2
#ifndef BENCH_AVX
#ifndef BENCH_SSE
#ifndef BENCH_POWER8
#ifndef BENCH_ARMV8
	  for (j=start; j< start+chunk; j++)
            c[j] = a[j];
#endif
#endif
#endif
#endif
#endif
#endif
        }
#endif
}

void tuned_STREAM_Scale(STREAM_TYPE scalar)
{
#ifdef USE_CUDA_HMM
	int thr_per_blk = CUDA_THREAD_PER_BLOCK;
	int blk_in_grid = ceil( float(STREAM_ARRAY_SIZE) / thr_per_blk );

  tuned_STREAM_Scale_cuda<<< blk_in_grid, thr_per_blk >>>(c, b);
  cudaDeviceSynchronize();
#elif defined(USE_SYCL_USM)
    tuned_STREAM_Scale_sycl(sycl_q, c, b);
    sycl_q.wait();
#else
#ifdef BENCH_AVX512
        __m512d vecscalar = _mm512_set1_pd(scalar);
#endif
#ifdef BENCH_AVX
        __m256d vecscalar = _mm256_set1_pd(scalar);
#endif
#ifdef BENCH_SSE
        __m128d vecscalar = _mm_set1_pd(scalar);
#endif
#ifdef BENCH_POWER8
        __attribute__((aligned(128))) STREAM_TYPE pumped_scalar[2] = {scalar, scalar};
        vector double vecscalar = vec_vsx_ld(0, pumped_scalar);
#endif
#ifdef BENCH_RV64
    size_t gvl = __riscv_vsetvl_e64m1(16);
    double vecscalar = scalar;
#endif

        #pragma omp parallel
        {
          ssize_t j;
#ifdef _OPENMP
          ssize_t chunk = STREAM_ARRAY_SIZE/omp_get_num_threads();
          ssize_t start = chunk*omp_get_thread_num();
#else
          ssize_t chunk = STREAM_ARRAY_SIZE;
          ssize_t start = 0;

#endif     

#ifdef BENCH_AVX512
	  for (j=start; j< start+chunk; j+=8)
            _mm512_stream_pd(&b[j], _mm512_mul_pd(vecscalar, _mm512_load_pd(&c[j])));
#endif 
#ifdef BENCH_AVX    
	  for (j=start; j< start+chunk; j+=4)
            _mm256_stream_pd(&b[j], _mm256_mul_pd(vecscalar, _mm256_load_pd(&c[j])));
#endif
#ifdef BENCH_SSE    
	  for (j=start; j< start+chunk; j+=2)
            _mm_stream_pd(&b[j], _mm_mul_pd(vecscalar, _mm_load_pd(&c[j])));
#endif
#ifdef BENCH_POWER8    
	  for (j=start; j< start+chunk; j+=2)
            vec_vsx_st(vec_mul(vecscalar, vec_vsx_ld(0, &c[j])), 0, &b[j]);
#endif
#ifdef BENCH_RV64    
	  for (j=start; j< start+chunk; j+=2)
            __riscv_vse64_v_f64m1(&b[j], __riscv_vfmul_vf_f64m1(__riscv_vle64_v_f64m1(&c[j], gvl), vecscalar, gvl), gvl);
#endif
#ifdef BENCH_ARMV8
        __asm__ __volatile__("mov x0, %0\n\t"  // b
                             "mov x1, %1\n\t"  // c
                             "mov x3, %2\n\t"  // scalar
                             "mov x4, %3\n\t"  // chunk
                             "ldr d6, [x3]\n\t"
                             "1:\n\t"                            
                             "ldr  d0, [x1]\n\t"
                             "ldr  d1, [x1,8]\n\t"
                             "fmul d0, d0, d6\n\t"
                             "fmul d1, d1, d6\n\t"
                             "stnp d0, d1, [x0]\n\t" 
                             "ldr  d2, [x1,16]\n\t"
                             "ldr  d3, [x1,24]\n\t"
                             "fmul d2, d2, d6\n\t"
                             "fmul d3, d3, d6\n\t"
                             "stnp d2, d3, [x0,16]\n\t" 
                             "add x0, x0, #32\n\t"
                             "add x1, x1, #32\n\t"
                             "sub x4, x4, #4\n\t"
                             "cbnz x4, 1b\n\t"
                        : : "r" (&b[start]), "r" (&c[start]), "r" (&scalar), "r" (chunk) : "x0","x1","x3","x4","d0","d1","d2","d3","d6");
#endif

#ifndef BENCH_AVX512
#ifndef BENCH_AVX2
#ifndef BENCH_AVX
#ifndef BENCH_SSE
#ifndef BENCH_POWER8
#ifndef BENCH_ARMV8
	  for (j=start; j< start+chunk; j++)
            b[j] = scalar * c[j];
#endif
#endif
#endif
#endif
#endif
#endif
        }
#endif
}

void tuned_STREAM_Add()
{
#ifdef USE_CUDA_HMM
	int thr_per_blk = CUDA_THREAD_PER_BLOCK;
	int blk_in_grid = ceil( float(STREAM_ARRAY_SIZE) / thr_per_blk );

  tuned_STREAM_Add_cuda<<< blk_in_grid, thr_per_blk >>>(a, b, c);
  cudaDeviceSynchronize();
#elif defined(USE_SYCL_USM)
    tuned_STREAM_Add_sycl(sycl_q, a, b, c);
    sycl_q.wait();
#else
        #pragma omp parallel
        {
          ssize_t j;
#ifdef _OPENMP
          ssize_t chunk = STREAM_ARRAY_SIZE/omp_get_num_threads();
          ssize_t start = chunk*omp_get_thread_num();
#else
          ssize_t chunk = STREAM_ARRAY_SIZE;
          ssize_t start = 0;
#endif     

#ifdef BENCH_AVX512  
	  for (j=start; j< start+chunk; j+=8)
           _mm512_stream_pd(&c[j], _mm512_add_pd(_mm512_load_pd(&a[j]), _mm512_load_pd(&b[j])));
#endif
#ifdef BENCH_AVX     
	  for (j=start; j< start+chunk; j+=4)
           _mm256_stream_pd(&c[j], _mm256_add_pd(_mm256_load_pd(&a[j]), _mm256_load_pd(&b[j])));
#endif
#ifdef BENCH_SSE     
	  for (j=start; j< start+chunk; j+=2)
           _mm_stream_pd(&c[j], _mm_add_pd(_mm_load_pd(&a[j]), _mm_load_pd(&b[j])));
#endif
#ifdef BENCH_SSE     
	  for (j=start; j< start+chunk; j+=2)
           _mm_stream_pd(&c[j], _mm_add_pd(_mm_load_pd(&a[j]), _mm_load_pd(&b[j])));
#endif
#ifdef BENCH_POWER8     
	  for (j=start; j< start+chunk; j+=2)
            vec_vsx_st(vec_add(vec_vsx_ld(0, &a[j]), vec_vsx_ld(0, &b[j])), 0, &c[j]);
#endif
#ifdef BENCH_RV64
    size_t gvl = __riscv_vsetvl_e64m1(16);
	  for (j=start; j< start+chunk; j+=2){
            __riscv_vse64_v_f64m1(&c[j], __riscv_vfadd_vv_f64m1(__riscv_vle64_v_f64m1((&a[j]), gvl), __riscv_vle64_v_f64m1((&b[j]), gvl), gvl), gvl);
    }
#endif
#ifdef BENCH_ARMV8    
        __asm__ __volatile__("mov x0, %0\n\t"  // a
                             "mov x1, %1\n\t"  // b
                             "mov x2, %2\n\t"  // c
                             "mov x4, %3\n\t"  // chunk
                             "1:\n\t"                            
                             "ldr  d0, [x0]\n\t"
                             "ldr  d1, [x1]\n\t"
                             "fadd  d0, d1, d0\n\t"
                             "ldr  d2, [x0,8]\n\t"
                             "ldr  d3, [x1,8]\n\t"
                             "fadd  d2, d3, d2\n\t"
                             "stnp d0, d2, [x2]\n\t"
                             "ldr  d4, [x0,16]\n\t"
                             "ldr  d5, [x1,16]\n\t"
                             "fadd  d4, d5, d4\n\t"
                             "ldr  d7, [x0,24]\n\t"
                             "ldr  d8, [x1,24]\n\t"
                             "fadd  d7, d8, d7\n\t"
                             "stnp d4, d7, [x2,16]\n\t" 
                             "add x0, x0, #32\n\t"
                             "add x1, x1, #32\n\t"
                             "add x2, x2, #32\n\t"
                             "sub x4, x4, #4\n\t"
                             "cbnz x4, 1b\n\t"
                        : : "r" (&a[start]), "r" (&b[start]), "r" (&c[start]), "r" (chunk) : "x0","x1","x2","x3","x4","d0","d1","d2","d3","d4","d5","d6","d7","d8");
#endif

#ifndef BENCH_AVX512
#ifndef BENCH_AVX2
#ifndef BENCH_AVX
#ifndef BENCH_SSE
#ifndef BENCH_POWER8
#ifndef BENCH_ARMV8
	  for (j=start; j< start+chunk; j++)
            c[j] = a[j] + b[j];
#endif
#endif
#endif
#endif
#endif
#endif
        }
#endif
}

void tuned_STREAM_Triad(STREAM_TYPE scalar)
{
#ifdef USE_CUDA_HMM
	int thr_per_blk = CUDA_THREAD_PER_BLOCK;
	int blk_in_grid = ceil( float(STREAM_ARRAY_SIZE) / thr_per_blk );

  tuned_STREAM_Triad_cuda<<< blk_in_grid, thr_per_blk >>>(a, b, c);
  cudaDeviceSynchronize();
#elif defined(USE_SYCL_USM)
    tuned_STREAM_Triad_sycl(sycl_q, a, b, c);
    sycl_q.wait();
#else
#ifdef BENCH_AVX512
        __m512d vecscalar = _mm512_set1_pd(scalar);
#endif
#ifdef BENCH_AVX
        __m256d vecscalar = _mm256_set1_pd(scalar);
#endif
#ifdef BENCH_SSE
        __m128d vecscalar = _mm_set1_pd(scalar);
#endif
#ifdef BENCH_POWER8
        __attribute__((aligned(128))) STREAM_TYPE pumped_scalar[2] = {scalar, scalar};
        vector double vecscalar = vec_vsx_ld(0, pumped_scalar);
#endif
#ifdef BENCH_RV64
        size_t gvl = __riscv_vsetvl_e64m1(16);
        double vecscalar = scalar;
#endif
        #pragma omp parallel
        {
          ssize_t j;
#ifdef _OPENMP
          ssize_t chunk = STREAM_ARRAY_SIZE/omp_get_num_threads();
          ssize_t start = chunk*omp_get_thread_num();
#else
          ssize_t chunk = STREAM_ARRAY_SIZE;
          ssize_t start = 0;
#endif      

#ifdef BENCH_AVX512    
	  for (j=start; j< start+chunk; j+=8) {
#ifdef USE_AVX512_PREFETCH
             _mm_prefetch( (void*) &b[j+64], _MM_HINT_T2 );
             _mm_prefetch( (void*) &c[j+64], _MM_HINT_T2 );
             _mm_prefetch( (void*) &b[j+16], _MM_HINT_T1 );
             _mm_prefetch( (void*) &c[j+16], _MM_HINT_T1 );
#endif
             _mm512_stream_pd(&a[j], _mm512_add_pd(_mm512_load_pd(&b[j]), _mm512_mul_pd(vecscalar, _mm512_load_pd(&c[j])) ) );
    }
#endif
#ifdef BENCH_AVX    
	  for (j=start; j< start+chunk; j+=4)
             _mm256_stream_pd(&a[j], _mm256_add_pd(_mm256_load_pd(&b[j]), _mm256_mul_pd(vecscalar, _mm256_load_pd(&c[j])) ) );
#endif
#ifdef BENCH_SSE    
	  for (j=start; j< start+chunk; j+=2)
             _mm_stream_pd(&a[j], _mm_add_pd(_mm_load_pd(&b[j]), _mm_mul_pd(vecscalar, _mm_load_pd(&c[j])) ) );
#endif
#ifdef BENCH_POWER8
	  for (j=start; j< start+chunk; j+=2)
            vec_vsx_st(vec_madd(vec_vsx_ld(0, &c[j]), vecscalar, vec_vsx_ld(0, &b[j])), 0, &a[j]);
#endif
#ifdef BENCH_RV64
	  for (j=start; j< start+chunk; j+=2)
             __riscv_vse64_v_f64m1(&a[j], __riscv_vfmadd_vf_f64m1(__riscv_vle64_v_f64m1(&c[j], gvl), vecscalar, __riscv_vle64_v_f64m1(&b[j], gvl), gvl), gvl);
#endif
#ifdef BENCH_ARMV8
        __asm__ __volatile__("mov x0, %0\n\t"  // a
                             "mov x1, %1\n\t"  // b
                             "mov x2, %2\n\t"  // c
                             "mov x3, %3\n\t"  // scalar
                             "mov x4, %4\n\t"  // chunk
                             "ldr d6, [x3]\n\t"
                             "1:\n\t"                            
                             "ldr  d0, [x1]\n\t"
                             "ldr  d1, [x2]\n\t"
                             "fmadd  d0, d6, d1, d0\n\t"
                             "ldr  d2, [x1,8]\n\t"
                             "ldr  d3, [x2,8]\n\t"
                             "fmadd  d2, d6, d3, d2\n\t"
                             "stnp d0, d2, [x0]\n\t"
                             "ldr  d4, [x1,16]\n\t"
                             "ldr  d5, [x2,16]\n\t"
                             "fmadd  d4, d6, d5, d4\n\t"
                             "ldr  d7, [x1,24]\n\t"
                             "ldr  d8, [x2,24]\n\t"
                             "fmadd  d7, d6, d8, d7\n\t"
                             "stnp d4, d7, [x0,16]\n\t" 
                             "add x0, x0, #32\n\t"
                             "add x1, x1, #32\n\t"
                             "add x2, x2, #32\n\t"
                             "sub x4, x4, #4\n\t"
                             "cbnz x4, 1b\n\t"
                        : : "r" (&a[start]), "r" (&b[start]), "r" (&c[start]), "r" (&scalar), "r" (chunk) : "x0","x1","x2","x3","x4","d0","d1","d2","d3","d4","d5","d6","d7","d8");
#endif

#ifndef BENCH_AVX512
#ifndef BENCH_AVX2
#ifndef BENCH_AVX
#ifndef BENCH_SSE
#ifndef BENCH_POWER8
#ifndef BENCH_ARMV8
	  for (j=start; j< start+chunk; j++)
            a[j] = b[j] + (scalar * c[j]);
#endif
#endif
#endif
#endif
#endif
#endif
          
        }
#endif
}
/* end of stubs for the "tuned" versions of the kernels */
#endif
