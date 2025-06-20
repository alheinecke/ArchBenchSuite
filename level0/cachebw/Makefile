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

TARGET = x86
CFLAGS = -O2 -fopenmp -fstrict-aliasing

ifeq ($(TARGET), x86)
CC = clang
EXE = cachebw.sse.exe cachebw.avx2.exe cachebw.avx512.exe cachebw.avx512bcast128.exe cachebw.avx512bcast256.exe
endif

ifeq ($(TARGET), armv8)
CC = clang
EXE = cachebw.armv8.exe
endif

ifeq ($(TARGET), rv64)
CC = clang
EXE = cachebw.rv64.exe
endif

ifeq ($(TARGET), ppc64le)
CC = clang
EXE = cachebw.ppc64le.exe
endif

all: $(EXE)

#cachebw.exe: cachebw.c ./../common/perf_counter_markers.c
#	$(CC) $(CFLAGS) ./../common/perf_counter_markers.c cachebw.c -o cachebw.exe

cachebw.sse.exe: cachebw.c
	$(CC) $(CFLAGS) -msse3 cachebw.c -o cachebw.sse.exe

cachebw.avx2.exe: cachebw.c
	$(CC) $(CFLAGS) -mavx2 cachebw.c -o cachebw.avx2.exe

cachebw.avx512.exe: cachebw.c
	$(CC) $(CFLAGS) -mavx512f -mavx512cd -mavx512dq -mavx512bw cachebw.c -o cachebw.avx512.exe

cachebw.avx512bcast128.exe: cachebw.c
	$(CC) $(CFLAGS) -mavx512f -mavx512cd -mavx512dq -mavx512bw cachebw.c -DAVX512_BCAST128 -o cachebw.avx512bcast128.exe

cachebw.avx512bcast256.exe: cachebw.c
	$(CC) $(CFLAGS) -mavx512f -mavx512cd -mavx512dq -mavx512bw cachebw.c -DAVX512_BCAST256 -o cachebw.avx512bcast256.exe

cachebw.armv8.exe: cachebw.c
	$(CC) $(CFLAGS) -march=armv8-a cachebw.c -o cachebw.armv8.exe

cachebw.rv64.exe: cachebw.c
	$(CC) $(CFLAGS) -march=rv64imfdv cachebw.c -o cachebw.rv64.exe

cachebw.ppc64le.exe: cachebw.c
	$(CC) $(CFLAGS) -mcpu=power8 -mvsx -mdirect-move -mpower8-fusion -mpower8-vector cachebw.c -o cachebw.ppc64le.exe

clean:
	rm -f cachebw.*.exe *.o
