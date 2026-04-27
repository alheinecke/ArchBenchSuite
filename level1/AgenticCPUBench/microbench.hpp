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

#ifndef MICROBENCH_HPP
#define MICROBENCH_HPP

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

/**
 * A single performance sample collected during a benchmark invocation.
 */
struct PerfSample {
    std::string bench_name;    /**< Name of the benchmark that produced this sample. */
    int    reps;               /**< Number of repetitions in this sample. */
    double monotonic_time_s;   /**< CLOCK_MONOTONIC at measurement start. */
    double elapsed_s;          /**< Measured total duration in seconds. */
    /** Benchmark-specific metric name/value pairs. */
    std::vector<std::pair<std::string, double>> metrics;
};

/**
 * Abstract base class for all micro-benchmarks.
 *
 * Derived classes must implement:
 *   setup_benchmark   – allocate resources and perform any warm-up.
 *   run_benchmark     – execute one iteration of the benchmark kernel.
 *   destroy_benchmark – release all allocated resources.
 */
class MicroBench {
public:
    virtual ~MicroBench() = default;

    void set_thread_id(int id) { thread_id_ = id; }
    int  get_thread_id() const { return thread_id_; }

    void set_bench_name(const std::string& name) { bench_name_ = name; }
    const std::string& get_bench_name() const { return bench_name_; }

    void set_reps_mult(int m) { reps_mult_ = m; }
    int  get_reps_mult() const { return reps_mult_; }

    /**
     * Allocate resources and warm up the benchmark.
     * @param size_bytes  Working-set size in bytes.
     */
    virtual void setup_benchmark(size_t size_bytes) = 0;

    /**
     * Run the benchmark kernel @p reps times, collect precise timing and
     * performance metrics into perf_data_.
     * @param reps  Number of timed repetitions to execute.
     */
    virtual void run_benchmark(int reps) = 0;

    /**
     * Release all resources acquired in setup_benchmark.
     */
    virtual void destroy_benchmark() = 0;

    /**
     * Access the collected performance samples (read-only).
     */
    const std::vector<PerfSample>& get_perf_data() const { return perf_data_; }

    /**
     * Merge samples from multiple benchmarks into a single per-thread CSV,
     * sorted by monotonic timestamp.
     * Filename: <tid>.csv
     *
     * The column set is the union of all metric names across benchmarks;
     * missing metrics are left empty.
     */
    static void dump_merged_perf_data(const std::vector<const MicroBench*>& benches)
    {
        if (benches.empty()) return;

        const int tid = benches[0]->get_thread_id();

        /* Collect all samples into one vector */
        std::vector<const PerfSample*> all;
        for (const auto* b : benches)
            for (const auto& s : b->get_perf_data())
                all.push_back(&s);

        if (all.empty()) return;

        /* Sort by monotonic timestamp */
        std::sort(all.begin(), all.end(),
                  [](const PerfSample* a, const PerfSample* b) {
                      return a->monotonic_time_s < b->monotonic_time_s;
                  });

        /* Build ordered union of all metric column names */
        std::vector<std::string> metric_cols;
        std::map<std::string, size_t> col_index;
        for (const auto* sp : all) {
            for (const auto& m : sp->metrics) {
                if (col_index.find(m.first) == col_index.end()) {
                    col_index[m.first] = metric_cols.size();
                    metric_cols.push_back(m.first);
                }
            }
        }

        char host[256];
        gethostname(host, 255);

        const std::string filename = std::to_string(tid) + ".csv";
        std::ofstream ofs(filename);
        if (!ofs.is_open()) return;

        /* Header */
        ofs << "host,thread_id,bench_name,reps,monotonic_time_s,elapsed_s";
        for (const auto& c : metric_cols)
            ofs << "," << c;
        ofs << "\n";

        ofs.precision(15);

        /* Data rows */
        for (const auto* sp : all) {
            ofs << host << ","
                << tid << ","
                << sp->bench_name << ","
                << sp->reps << ","
                << sp->monotonic_time_s << ","
                << sp->elapsed_s;

            /* Build a lookup for this sample's metrics */
            std::map<std::string, double> mmap;
            for (const auto& m : sp->metrics)
                mmap[m.first] = m.second;

            for (const auto& c : metric_cols) {
                auto it = mmap.find(c);
                if (it != mmap.end())
                    ofs << "," << it->second;
                else
                    ofs << ",";
            }
            ofs << "\n";
        }
    }

protected:
    int         thread_id_  = 0;
    int         reps_mult_  = 1;
    std::string bench_name_;
    std::vector<PerfSample> perf_data_;

    static inline double sec(struct timeval start, struct timeval end) {
        return static_cast<double>(
                   (end.tv_sec  * 1000000 + end.tv_usec) -
                   (start.tv_sec * 1000000 + start.tv_usec)) / 1.0e6;
    }

    /**
     * Capture monotonic timestamp.
     */
    static inline double capture_monotonic() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<double>(ts.tv_sec) +
               static_cast<double>(ts.tv_nsec) / 1.0e9;
    }
};

#endif /* MICROBENCH_HPP */
