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

#CC = g++
#CFLAGS = -Ofast -static -ansi-alias -mavx -mfma4 -fopenmp -ftree-vectorize -fprefetch-loop-arrays -funroll-all-loops -ffast-math -march=bdver1
#CFLAGS = -Ofast -static -ansi-alias -mavx -mfma4 -fopenmp -ftree-vectorize -fprefetch-loop-arrays -funroll-all-loops -ffast-math ${ACML_INC} ${ACML_LIB}
CC = icc
CFLAGS = -Ofast -static -ansi-alias -mavx -openmp -unroll-loops ${ACML_INC} /lrz/sys/libraries/acml/5.2.0/ifort64_fma4/lib/libacml.a -lgfortran
CFLAGS = -Ofast -static -ansi-alias -mavx -openmp -unroll-loops ${ACML_INC} ${ACML_SHLIB} -L/lrz/sys/intel/ifort_131_163/composer_xe_2013.3.163/compiler/lib/intel64 -lifcore
CFLAGS = -Ofast -ansi-alias -mavx -L${MKLROOT}/lib/intel64/ -mkl -unroll-loop -D__MKL -static-intel -qopenmp-link=static

all: sgemm.exe

sgemm.exe: test_sgemm.cpp
	$(CC) test_sgemm.cpp $(CFLAGS) -o sgemm.exe

clean:
	rm -f sgemm.exe *.o
