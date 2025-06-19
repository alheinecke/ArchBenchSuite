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
#CFLAGS = -O2 -fopenmp -DNTIMES=100 -DSTREAM_ARRAY_SIZE=199808000 -mcpu=power8 -fstrict-aliasing
#CFLAGS = -O2 -fopenmp -DNTIMES=100 -DSTREAM_ARRAY_SIZE=199808000 -march=armv8-a -fstrict-aliasing
#CFLAGS = -O2 -fopenmp -DNTIMES=100 -DSTREAM_ARRAY_SIZE=199808000 -msse3 -fstrict-aliasing
CFLAGS = -O2 -fopenmp -DNTIMES=100 -DSTREAM_ARRAY_SIZE=199808000 -mavx2 -fstrict-aliasing
#CFLAGS = -O2 -fopenmp -DNTIMES=100 -DSTREAM_ARRAY_SIZE=199808000 -mavx512f -mavx512cd -fstrict-aliasing
#CC = icc
#CFLAGS = -O3 -xCOMMON-AVX512 -qopenmp -DNTIMES=10000 -DSTREAM_ARRAY_SIZE=199808000 -qopt-streaming-cache-evict=0 -qopt-streaming-stores always -qopt-prefetch-distance=64,8 -qopenmp-link=static
#CFLAGS = -O3 -xCOMMON-AVX512 -qopenmp -DNTIMES=10000 -DSTREAM_ARRAY_SIZE=199808000 -qopt-streaming-cache-evict=0 -qopt-streaming-stores always -qopt-prefetch=5 -qopt-prefetch-distance=448,32 -qopenmp-link=static

CC=clang
CFLAGS=-O3 -mavx512f -mavx512cd -mavx512bw -mavx512dq -fopenmp -DBENCH_AVX512 -DNTIMES=100 -DSTREAM_ARRAY_SIZE=299808000 -fstrict-aliasing
CFLAGS=-O3 -mavx2 -fopenmp -DNTIMES=100 -DSTREAM_ARRAY_SIZE=299808000 -fstrict-aliasing

all: readbw.exe

#readbw.exe: readbw.c
#	$(CC) $(CFLAGS) readbw.c -o readbw.exe

readbw.exe: readbw.c ./../common/perf_counter_markers.c
	$(CC) $(CFLAGS) readbw.c ./../common/perf_counter_markers.c -o readbw.exe

clean:
	rm -f readbw.exe *.o
