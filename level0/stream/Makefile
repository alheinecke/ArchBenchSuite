#############################################################################
# Copyright (c) 2013-2018, Alexander Heinecke                               #
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

CC = gcc
TARGET=sse
SOURCE = stream.c

ifeq ($(TARGET), sse)
CFLAGS=-O2 -fopenmp -DBENCH_SSE -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -msse3 -fstrict-aliasing
EXE=stream_sse
endif

ifeq ($(TARGET), avx)
CFLAGS=-O2 -fopenmp -DBENCH_AVX -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -mavx -fstrict-aliasing
CFLAGS=-O2 -mavx -fopenmp -DBENCH_AVX -DNTIMES=100 -DSTREAM_ARRAY_SIZE=1442119680 -fstrict-aliasing
EXE=stream_avx
endif

ifeq ($(TARGET), avx2)
CC=clang
#CFLAGS=-O2 -mavx -qopenmp -DBENCH_AVX512 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores always -opt-prefetch-distance=64,8 -static-intel
CFLAGS=-O2 -mavx2 -mfma -fopenmp -DBENCH_AVX2 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=1442119680 -fstrict-aliasing
EXE=stream_avx2
endif

ifeq ($(TARGET), avx512)
CC=clang
#CFLAGS=-O2 -xCOMMON-AVX512 -qopenmp -DBENCH_AVX512 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores always -opt-prefetch-distance=64,8 -static-intel
#CFLAGS=-O2 -xCOMMON-AVX512 -qopenmp -DBENCH_AVX512 -DNTIMES=1000 -DSTREAM_ARRAY_SIZE=43008000 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores always -opt-prefetch-distance=64,8 -static-intel
# 68 cores
#CFLAGS=-O2 -xCOMMON-AVX512 -qopenmp -DBENCH_AVX512 -DNTIMES=1000 -DSTREAM_ARRAY_SIZE=73740288 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores always -opt-prefetch-distance=64,8 -static-intel
#CFLAGS=-O2 -xCOMMON-AVX512 -qopenmp -DBENCH_AVX512 -DNTIMES=10000 -DSTREAM_ARRAY_SIZE=2752512 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores never -opt-prefetch-distance=64,8 -static-intel
CFLAGS=-O2 -xCOMMON-AVX512 -qopenmp -DBENCH_AVX512 -DNTIMES=1000 -DSTREAM_ARRAY_SIZE=64512000 -fstrict-aliasing -qopt-streaming-cache-evict=0 -qopt-streaming-stores always -qopt-prefetch-distance=64,8 -static-intel
CFLAGS=-O2 -mavx512f -mavx512cd -mavx512bw -mavx512dq -fopenmp -DBENCH_AVX512 -DNTIMES=10 -DSTREAM_ARRAY_SIZE=64512000 -fstrict-aliasing
CFLAGS=-O3 -mavx512f -mavx512cd -mavx512bw -mavx512dq -fopenmp -DBENCH_AVX512 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=64512000 -fstrict-aliasing
CFLAGS=-O3 -mavx512f -mavx512cd -mavx512bw -mavx512dq -fopenmp -DBENCH_AVX512 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -fstrict-aliasing
CFLAGS=-O3 -mavx512f -mavx512cd -mavx512bw -mavx512dq -fopenmp -DBENCH_AVX512 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=715833344 -fstrict-aliasing
CFLAGS=-O3 -mavx512f -mavx512cd -mavx512bw -mavx512dq -fopenmp -DBENCH_AVX512 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=1431633920 -fstrict-aliasing
#CFLAGS=-O2 -xCOMMON-AVX512 -qopenmp -DBENCH_AVX512 -DNTIMES=1000 -DSTREAM_ARRAY_SIZE=64512000 -fstrict-aliasing -qopt-streaming-cache-evict=0 -qopt-streaming-stores always -qopt-prefetch=5 -qopt-prefetch-distance=448,32 -static-intel
EXE=stream_avx512
endif

ifeq ($(TARGET), cuda)
CC=nvcc
# 16GB
CFLAGS=-O2 -DUSE_CUDA_HMM -DNTIMES=100 -DSTREAM_ARRAY_SIZE=715833344 -allow-unsupported-compiler
# 32GB
#CFLAGS=-O2 -DUSE_CUDA_HMM -DNTIMES=5 -DSTREAM_ARRAY_SIZE=1431655424 -allow-unsupported-compiler
EXE=stream_cuda
SOURCE=stream.cu
endif

ifeq ($(TARGET), cuda_host)
CC=nvcc
# 16GB
#CFLAGS=-O2 -DUSE_CUDA_HMM -DUSE_CUDA_HOST_ALLOC -DNTIMES=10 -DSTREAM_ARRAY_SIZE=715833344 -allow-unsupported-compiler
# 32GB
CFLAGS=-O2 -DUSE_CUDA_HMM -DUSE_CUDA_HOST_ALLOC -DNTIMES=5 -DSTREAM_ARRAY_SIZE=1431655424 -allow-unsupported-compiler
EXE=stream_cuda_host
SOURCE=stream.cu
endif

ifeq ($(TARGET), cuda_device)
CC=nvcc
# 16GB
CFLAGS=-O2 -DUSE_CUDA_HMM -DUSE_CUDA_DEVICE_ALLOC -DNTIMES=100 -DSTREAM_ARRAY_SIZE=715833344 -allow-unsupported-compiler
EXE=stream_cuda_device
SOURCE=stream.cu
endif

ifeq ($(TARGET), sycl)
CC=icpx
# 4GB
#CFLAGS=-O2 -fsycl -DUSE_SYCL_USM -DNTIMES=10 -DSTREAM_ARRAY_SIZE=178958336
# 8GB
CFLAGS=-O2 -fsycl -DUSE_SYCL_USM -DNTIMES=100 -DSTREAM_ARRAY_SIZE=357916672
# 16GB
#CFLAGS=-O2 -fsycl -DUSE_SYCL_USM -DNTIMES=5 -DSTREAM_ARRAY_SIZE=715833344
# 32GB
#CFLAGS=-O2 -fsycl -DUSE_SYCL_USM -DNTIMES=5 -DSTREAM_ARRAY_SIZE=1431655424
EXE=stream_sycl
SOURCE=stream.c
endif

ifeq ($(TARGET), sycl_host)
CC=icpx
# 8GB
CFLAGS=-O2 -fsycl -DUSE_SYCL_USM -DUSE_SYCL_HOST_ALLOC -DNTIMES=10 -DSTREAM_ARRAY_SIZE=357916672
# 16GB
#CFLAGS=-O2 -fsycl -DUSE_SYCL_USM -DUSE_SYCL_HOST_ALLOC -DNTIMES=5 -DSTREAM_ARRAY_SIZE=715833344
# 32GB
#CFLAGS=-O2 -fsycl -DUSE_SYCL_USM -DUSE_SYCL_HOST_ALLOC -DNTIMES=5 -DSTREAM_ARRAY_SIZE=1431655424
EXE=stream_sycl_host
SOURCE=stream.c
endif

ifeq ($(TARGET), sycl_device)
CC=icpx
# 8GB
CFLAGS=-O2 -fsycl -DUSE_SYCL_USM -DUSE_SYCL_DEVICE_ALLOC -DNTIMES=10 -DSTREAM_ARRAY_SIZE=357916672
EXE=stream_sycl_device
SOURCE=stream.c
endif

ifeq ($(TARGET), sycl_shared)
CC=icpx
# 8GB
CFLAGS=-O2 -fsycl -DUSE_SYCL_USM -DUSE_SYCL_SHARED_ALLOC -DNTIMES=10 -DSTREAM_ARRAY_SIZE=357916672
EXE=stream_sycl_shared
SOURCE=stream.c
endif

ifeq ($(TARGET), armv8)
CFLAGS=-O2 -fopenmp -DBENCH_ARMV8 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -march=armv8-a -fstrict-aliasing
EXE=stream_armv8
endif 

ifeq ($(TARGET), rv64)
CFLAGS=-O2 -DBENCH_RV64 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -march=rv64imfdv -fopenmp 
EXE=stream_rv64
endif

ifeq ($(TARGET), power8)
CFLAGS=-O2 -fopenmp -DBENCH_POWER8 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -mcpu=power8 -mvsx -mdirect-move -mpower8-fusion -mpower8-vector -fstrict-aliasing
EXE=stream_power8
endif 

default: stream.c
ifeq ($(TARGET), cuda)
	cp stream.c stream.cu
endif
ifeq ($(TARGET), cuda_host)
	cp stream.c stream.cu
endif
ifeq ($(TARGET), cuda_device)
	cp stream.c stream.cu
endif
	$(CC) $(CFLAGS) -o $(EXE) $(SOURCE)

clean:
	rm -f stream_* *.o
