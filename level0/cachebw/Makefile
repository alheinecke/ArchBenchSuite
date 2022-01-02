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
#CFLAGS = -O2 -fopenmp -mcpu=power8 -mvsx -mdirect-move -mpower8-fusion -mpower8-vector -fstrict-aliasing
CFLAGS = -O2 -fopenmp -march=armv8-a -fstrict-aliasing
#CFLAGS = -O2 -fopenmp -mavx2 -fstrict-aliasing
#CFLAGS = -O2 -fopenmp -msse3 -fstrict-aliasing
#CFLAGS = -O2 -fopenmp -msse3 -fstrict-aliasing
CFLAGS = -O2 -msse -fstrict-aliasing
#CC = icc
#CFLAGS = -O3 -xHost -qopenmp
#CFLAGS = -O3 -xMIC-AVX512 -qopenmp
#CFLAGS = -O3 -xCOMMON-AVX512 -qopenmp
#CFLAGS = -O3 -xCORE-AVX2 -qopenmp

all: cachebw.exe

#cachebw.exe: cachebw.c ./../common/perf_counter_markers.c
#	$(CC) $(CFLAGS) ./../common/perf_counter_markers.c cachebw.c -o cachebw.exe

cachebw.exe: cachebw.c
	$(CC) $(CFLAGS) cachebw.c -o cachebw.exe

clean:
	rm -f cachebw.exe *.o
