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

ifeq ($(TARGET), sse)
CFLAGS=-O2 -fopenmp -DBENCH_SSE -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -msse3 -fstrict-aliasing
EXE=stream_sse
endif

ifeq ($(TARGET), avx)
CFLAGS=-O2 -fopenmp -DBENCH_AVX -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -mavx -fstrict-aliasing
EXE=stream_avx
endif

ifeq ($(TARGET), avx2)
CC=icc
#CFLAGS=-O2 -mavx -qopenmp -DBENCH_AVX512 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores always -opt-prefetch-distance=64,8 -static-intel
CFLAGS=-O2 -mavx -qopenmp -DBENCH_AVX512 -DNTIMES=1000 -DSTREAM_ARRAY_SIZE=43008000 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores always -opt-prefetch-distance=64,8 -static-intel
CFLAGS=-O2 -mavx -qopenmp -DBENCH_AVX2 -DNTIMES=1000 -DSTREAM_ARRAY_SIZE=73728000 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores always -opt-prefetch-distance=64,8 -static-intel
#CFLAGS=-O2 -mavx -qopenmp -DBENCH_AVX512 -DNTIMES=1000 -DSTREAM_ARRAY_SIZE=64512000 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores always -opt-prefetch-distance=64,8 -static-intel
EXE=stream_avx2
endif

ifeq ($(TARGET), avx512)
CC=icc
#CFLAGS=-O2 -xCOMMON-AVX512 -qopenmp -DBENCH_AVX512 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores always -opt-prefetch-distance=64,8 -static-intel
#CFLAGS=-O2 -xCOMMON-AVX512 -qopenmp -DBENCH_AVX512 -DNTIMES=1000 -DSTREAM_ARRAY_SIZE=43008000 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores always -opt-prefetch-distance=64,8 -static-intel
# 68 cores
#CFLAGS=-O2 -xCOMMON-AVX512 -qopenmp -DBENCH_AVX512 -DNTIMES=1000 -DSTREAM_ARRAY_SIZE=73740288 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores always -opt-prefetch-distance=64,8 -static-intel
#CFLAGS=-O2 -xCOMMON-AVX512 -qopenmp -DBENCH_AVX512 -DNTIMES=10000 -DSTREAM_ARRAY_SIZE=2752512 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores never -opt-prefetch-distance=64,8 -static-intel
CFLAGS=-O2 -xCOMMON-AVX512 -qopenmp -DBENCH_AVX512 -DNTIMES=1000 -DSTREAM_ARRAY_SIZE=64512000 -fstrict-aliasing -opt-streaming-cache-evict=0 -opt-streaming-stores always -opt-prefetch-distance=64,8 -static-intel
EXE=stream_avx512
endif

ifeq ($(TARGET), armv8)
CFLAGS=-O2 -fopenmp -DBENCH_ARMV8 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -march=armv8-a -fstrict-aliasing
EXE=stream_armv8
endif 

ifeq ($(TARGET), power8)
CFLAGS=-O2 -fopenmp -DBENCH_POWER8 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=73728000 -mcpu=power8 -mvsx -mdirect-move -mpower8-fusion -mpower8-vector -fstrict-aliasing
EXE=stream_power8
endif 

default: stream.c
	$(CC) $(CFLAGS) -o $(EXE) stream.c

clean:
	rm -f stream_* *.o
