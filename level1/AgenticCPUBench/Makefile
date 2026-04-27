#############################################################################
# Copyright (c) 2026, Alexander Heinecke                                    #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions        #
# are met:                                                                  #
# 1. Redistributions of source code must retain the above copyright         #
#    notice, this list of conditions and the following disclaimer.          #
# 2. Redistributions in binary form must reproduce the above copyright      #
#    notice, this list of conditions and the following disclaimer in the    #
#    documentation and/or other materials provided with the distribution.   #
# 3. Neither the name of the copyright holder nor the names of its          #
#    contributors may be used to endorse or promote products derived        #
#    from this software without specific prior written permission.          #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#############################################################################

CXX = g++
#CXX = icpc
#CXX = clang++

# libxsmm: cloned into ./libxsmm and built from source.
# Override LIBXSMM_ROOT on the command line to use a pre-installed copy:
#   make LIBXSMM_ROOT=/path/to/libxsmm/install
LIBXSMM_REPO  = https://github.com/libxsmm/libxsmm.git
LIBXSMM_ROOT ?= $(CURDIR)/libxsmm
LIBXSMM_LIB   = $(LIBXSMM_ROOT)/lib/libxsmm.a

# ISA selection: avx2 (default), avx512, aarch64.
#   make ISA=avx2      (default)
#   make ISA=avx512
#   make ISA=aarch64
ISA ?= avx2

CXXFLAGS_COMMON  = -O3 -fstrict-aliasing -std=c++11 -fopenmp -I$(LIBXSMM_ROOT)/include
CXXFLAGS_AVX2    = $(CXXFLAGS_COMMON) -mavx2
CXXFLAGS_AVX512  = $(CXXFLAGS_COMMON) -mavx512f -mavx512cd -mavx512bw -mavx512dq
CXXFLAGS_AARCH64 = $(CXXFLAGS_COMMON) -march=armv8-a

ifeq ($(ISA),avx512)
  CXXFLAGS = $(CXXFLAGS_AVX512)
else ifeq ($(ISA),aarch64)
  CXXFLAGS = $(CXXFLAGS_AARCH64)
else
  CXXFLAGS = $(CXXFLAGS_AVX2)
endif

LDFLAGS = -L$(LIBXSMM_ROOT)/lib -lxsmm -ldl -lpthread -lm -fopenmp

all: all_x86

# ---------------------------------------------------------------------------
# libxsmm: clone from GitHub if not present, then build
# ---------------------------------------------------------------------------

# Clone step – produces the .git directory
$(LIBXSMM_ROOT)/.git:
	git clone $(LIBXSMM_REPO) $(LIBXSMM_ROOT)

# Build step – produces the static library; depends on the clone
$(LIBXSMM_LIB): $(LIBXSMM_ROOT)/.git
	$(MAKE) -C $(LIBXSMM_ROOT) realclean
	$(MAKE) -C $(LIBXSMM_ROOT) -j16

.PHONY: libxsmm-build
libxsmm-build: $(LIBXSMM_LIB)

# All .cpp files in this directory contribute an object file
SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)

# ISA-suffixed object files: foo.cpp -> foo_avx2.o / foo_avx512.o
OBJS_ISA := $(SRCS:.cpp=_$(ISA).o)

# Header-only benchmarks and the abstract base are treated as dependencies
# for every object so that changes to any header trigger a recompile.
HDRS := $(wildcard *.hpp)

# Pattern rules: compile each .cpp to an ISA-suffixed .o
# Order-only dep on $(LIBXSMM_LIB) ensures libxsmm is cloned and built
# (headers present) before any of our sources are compiled.
%_avx2.o: %.cpp $(HDRS) | $(LIBXSMM_LIB)
	$(CXX) $(CXXFLAGS_AVX2) -c $< -o $@

%_avx512.o: %.cpp $(HDRS) | $(LIBXSMM_LIB)
	$(CXX) $(CXXFLAGS_AVX512) -c $< -o $@

%_aarch64.o: %.cpp $(HDRS) | $(LIBXSMM_LIB)
	$(CXX) $(CXXFLAGS_AARCH64) -c $< -o $@

AgenticCPUBench_$(ISA).exe: $(OBJS_ISA)
	$(CXX) $(OBJS_ISA) -o $@ $(LDFLAGS)

# ---------------------------------------------------------------------------
# all_x86: build both avx2 and avx512 variants
# ---------------------------------------------------------------------------
.PHONY: all_x86
all_x86:
	$(MAKE) ISA=avx2   AgenticCPUBench_avx2.exe
	$(MAKE) ISA=avx512 AgenticCPUBench_avx512.exe

clean:
	rm -f AgenticCPUBench_avx2.exe AgenticCPUBench_avx512.exe AgenticCPUBench_aarch64.exe \
	      $(foreach s,$(SRCS),$(s:.cpp=_avx2.o) $(s:.cpp=_avx512.o) $(s:.cpp=_aarch64.o))

# distclean also removes the cloned libxsmm source tree
.PHONY: distclean
distclean: clean
	rm -rf $(LIBXSMM_ROOT)
