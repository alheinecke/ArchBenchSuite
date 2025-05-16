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

CC=g++
TARGET=sse
OPENMP=0
CFLAGS:=-O2 -Wall
EXE=""

ifeq ($(TARGET), ssecpm)
CFLAGS:=$(CFLAGS) -DBENCH_CPMSSE -msse
EXE=peak_ssecpm
endif

ifeq ($(TARGET), ssecpm_s)
CFLAGS:=$(CFLAGS) -DBENCH_CPMSSE_SCALAR -msse
EXE=peak_ssecpm_s
endif

ifeq ($(TARGET), sseslm)
CFLAGS:=$(CFLAGS) -DBENCH_SLMSSE -msse3
EXE=peak_sseslm
endif

ifeq ($(TARGET), sse)
CFLAGS:=$(CFLAGS) -DBENCH_SSE -msse3
EXE=peak_sse
endif

ifeq ($(TARGET), avx)
CFLAGS:=$(CFLAGS) -DBENCH_AVX -mavx
EXE=peak_avx
endif

ifeq ($(TARGET), avx2)
CFLAGS:=$(CFLAGS) -DBENCH_AVX2 -mavx2
EXE=peak_avx2
endif

ifeq ($(TARGET), avx512)
CC=icpc
CFLAGS:=$(CFLAGS) -DBENCH_AVX512 -xCOMMON-AVX512
EXE=peak_avx512
endif

ifeq ($(TARGET), avx512qfma)
CC=icpc
CFLAGS:=$(CFLAGS) -DBENCH_AVX512 -DBENCH_AVX512_QFMA -xCOMMON-AVX512 -static-intel
EXE=peak_avx512qfma
endif

ifeq ($(TARGET), armv8)
CFLAGS:=$(CFLAGS) -DBENCH_ARMV8
EXE=peak_armv8
endif 

ifeq ($(TARGET), power8)
CFLAGS:=$(CFLAGS) -DBENCH_POWER8
EXE=peak_power8
endif 

ifeq ($(TARGET), rv64)
CFLAGS:=$(CFLAGS) -DBENCH_RV64 -march=rv64imfdv
EXE=peak_rv64
endif 

ifeq ($(OPENMP), 1)
CFLAGS:=$(CFLAGS) -fopenmp
endif

default: peak.cpp *.hpp
	$(CC) $(CFLAGS) -o $(EXE) peak.cpp

clean:
	rm -f peak_*
