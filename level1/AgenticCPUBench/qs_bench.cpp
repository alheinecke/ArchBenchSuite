/******************************************************************************
** Copyright (c) 2026, Alexander Heinecke                                    **
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

#include "qs_bench.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>

/* -------------------------------------------------------------------------
 * Quicksort kernel
 *
 * Characteristics:
 *   - Median-of-three pivot selection (avoids O(n^2) on sorted/reverse input)
 *   - Insertion sort below INSERTION_THRESHOLD to amortise recursion cost
 *   - Tail-call optimisation: recurse on the smaller partition, loop on
 *     the larger – bounds stack depth to O(log n)
 * -------------------------------------------------------------------------*/

static constexpr ptrdiff_t INSERTION_THRESHOLD = 16;

static inline void insertion_sort(int64_t* arr, ptrdiff_t lo, ptrdiff_t hi) {
    for (ptrdiff_t i = lo + 1; i <= hi; ++i) {
        const int64_t key = arr[i];
        ptrdiff_t j = i - 1;
        while (j >= lo && arr[j] > key) {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

static inline int64_t median3(int64_t a, int64_t b, int64_t c) {
    /* Returns the median of three values without branching on equality */
    if (a < b) {
        if (b < c) return b;          /* a < b < c */
        return (a < c) ? c : a;       /* a < c <= b  or  c <= a < b */
    } else {
        if (a < c) return a;          /* b <= a < c */
        return (b < c) ? c : b;       /* b < c <= a  or  c <= b <= a */
    }
}

static void quicksort(int64_t* arr, ptrdiff_t lo, ptrdiff_t hi) {
    while (lo < hi) {
        /* Fall back to insertion sort for tiny partitions */
        if (hi - lo < INSERTION_THRESHOLD) {
            insertion_sort(arr, lo, hi);
            return;
        }

        /* Median-of-three pivot: lo, mid, hi */
        const ptrdiff_t mid = lo + (hi - lo) / 2;
        const int64_t   pivot = median3(arr[lo], arr[mid], arr[hi]);

        /* Hoare-like two-pointer partition */
        ptrdiff_t i = lo;
        ptrdiff_t j = hi;
        while (i <= j) {
            while (arr[i] < pivot) ++i;
            while (arr[j] > pivot) --j;
            if (i <= j) {
                const int64_t tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
                ++i;
                --j;
            }
        }

        /* Recurse on the smaller half; iterate on the larger (limits stack depth) */
        if (j - lo < hi - i) {
            quicksort(arr, lo, j);
            lo = i;
        } else {
            quicksort(arr, i, hi);
            hi = j;
        }
    }
}

/* -------------------------------------------------------------------------
 * Helper
 * -------------------------------------------------------------------------*/

/* -------------------------------------------------------------------------
 * QsBench implementation
 * -------------------------------------------------------------------------*/

QsBench::QsBench() = default;

QsBench::~QsBench() {
    destroy_benchmark();
}

void QsBench::setup_benchmark(size_t size_bytes) {
    size_bytes_ = size_bytes;
    n_          = size_bytes / sizeof(int64_t);

    data_ = static_cast<int64_t*>(malloc(n_ * sizeof(int64_t)));
    ref_  = static_cast<int64_t*>(malloc(n_ * sizeof(int64_t)));

    if (!data_ || !ref_) {
        std::cerr << "QsBench::setup_benchmark: memory allocation failed." << std::endl;
        destroy_benchmark();
        return;
    }

    /* Initialise ref_ with values 0 … n-1, then Fisher-Yates shuffle */
    for (size_t i = 0; i < n_; ++i)
        ref_[i] = static_cast<int64_t>(i);

    srand(42);
    for (size_t i = n_ - 1; i > 0; --i) {
        const size_t j = static_cast<size_t>(rand()) % (i + 1);
        const int64_t tmp = ref_[i];
        ref_[i] = ref_[j];
        ref_[j] = tmp;
    }

    /* Warm-up: one sort pass to prime caches and branch predictors */
    memcpy(data_, ref_, n_ * sizeof(int64_t));
    quicksort(data_, 0, static_cast<ptrdiff_t>(n_) - 1);
}

void QsBench::run_benchmark(int reps) {
    reps *= reps_mult_;
    if (!data_ || !ref_) {
        std::cerr << "QsBench::run_benchmark: benchmark not initialised." << std::endl;
        return;
    }

    struct timeval t_start, t_end;

    const double mono_ts = capture_monotonic();

    gettimeofday(&t_start, NULL);
    for (int r = 0; r < reps; ++r) {
        /* Restore unsorted data before each sort */
        memcpy(data_, ref_, n_ * sizeof(int64_t));

        quicksort(data_, 0, static_cast<ptrdiff_t>(n_) - 1);
    }
    gettimeofday(&t_end, NULL);

    const double t = sec(t_start, t_end);

    const double melems_per_s = static_cast<double>(n_) * reps / t / 1.0e6;

    PerfSample s;
    s.bench_name       = bench_name_;
    s.reps             = reps;
    s.monotonic_time_s = mono_ts;
    s.elapsed_s        = t;
    s.metrics.push_back({"array_size_MB", size_bytes_ / (1024.0*1024.0)});
    s.metrics.push_back({"n_elements", static_cast<double>(n_)});
    s.metrics.push_back({"Melements_per_s", melems_per_s});
    perf_data_.push_back(s);
}

void QsBench::destroy_benchmark() {
    free(data_); data_ = nullptr;
    free(ref_);  ref_  = nullptr;
    n_          = 0;
    size_bytes_ = 0;
}
