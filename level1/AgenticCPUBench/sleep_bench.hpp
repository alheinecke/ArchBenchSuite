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

#ifndef SLEEP_BENCH_HPP
#define SLEEP_BENCH_HPP

#include "microbench.hpp"

#include <unistd.h>

/**
 * SleepBench – idle pseudo-benchmark.
 *
 * Each repetition sleeps for exactly 0.15 seconds using POSIX usleep().
 * Useful as a stand-in "idle" slot in the random benchmark round to
 * study system behaviour (power, frequency scaling, etc.) between
 * active benchmark runs.
 *
 * setup_benchmark / destroy_benchmark are no-ops.
 */
class SleepBench : public MicroBench {
public:
    void setup_benchmark(size_t /*size_bytes*/) override {}

    void run_benchmark(int reps) override {
        reps *= reps_mult_;
        static constexpr useconds_t SLEEP_US = 150000;  /* 0.15 s */

        const double mono_ts = capture_monotonic();

        struct timeval t_start, t_end;
        gettimeofday(&t_start, NULL);
        for (int r = 0; r < reps; ++r) {
            usleep(SLEEP_US);
        }
        gettimeofday(&t_end, NULL);

        const double elapsed =
            static_cast<double>(
                (t_end.tv_sec  * 1000000LL + t_end.tv_usec) -
                (t_start.tv_sec * 1000000LL + t_start.tv_usec)) / 1.0e6;

        PerfSample s;
        s.bench_name       = bench_name_;
        s.reps             = reps;
        s.monotonic_time_s = mono_ts;
        s.elapsed_s        = elapsed;
        s.metrics.push_back({"sleep_s", static_cast<double>(SLEEP_US) / 1.0e6});
        perf_data_.push_back(s);
    }

    void destroy_benchmark() override {}
};

#endif /* SLEEP_BENCH_HPP */
