/******************************************************************************
** Copyright (c) 2013-2018, Alexander Heinecke                               **
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

void gflops_double_qfma(double* data) {
  std::cout << "QFMA is not available on this architecture" << std::endl;
}

void gflops_double_fma(double* data) {
  std::cout << "FMA is not available on this architecture" << std::endl;
}

void gflops_double_mul(double* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "movupd (%%r8),  %%xmm0\n\t"
                         "movupd (%%r8),  %%xmm1\n\t"
                         "movupd (%%r8),  %%xmm2\n\t"
                         "movupd (%%r8),  %%xmm3\n\t"
                         "movupd (%%r8),  %%xmm4\n\t"
                         "movupd (%%r8),  %%xmm5\n\t"
                         "movupd (%%r8),  %%xmm6\n\t"
                         "movupd (%%r8),  %%xmm7\n\t"
                         "movupd (%%r8),  %%xmm8\n\t"
                         "movupd (%%r8),  %%xmm9\n\t"
                         "movupd (%%r8), %%xmm10\n\t"
                         "movupd (%%r8), %%xmm11\n\t"
                         "movupd (%%r8), %%xmm12\n\t"
                         "movupd (%%r8), %%xmm13\n\t"
                         "movupd (%%r8), %%xmm14\n\t"
                         "movupd (%%r8), %%xmm15\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "mulpd  %%xmm0,  %%xmm0\n\t"
                         "mulpd  %%xmm1,  %%xmm1\n\t"
                         "mulpd  %%xmm2,  %%xmm2\n\t"
                         "mulpd  %%xmm3,  %%xmm3\n\t"
                         "mulpd  %%xmm4,  %%xmm4\n\t"
                         "mulpd  %%xmm5,  %%xmm5\n\t"
                         "mulpd  %%xmm6,  %%xmm6\n\t"
                         "mulpd  %%xmm7,  %%xmm7\n\t"
                         "mulpd  %%xmm8,  %%xmm8\n\t"
                         "mulpd  %%xmm9,  %%xmm9\n\t"
                         "mulpd %%xmm10, %%xmm10\n\t"
                         "mulpd %%xmm11, %%xmm11\n\t"
                         "mulpd %%xmm12, %%xmm12\n\t"
                         "mulpd %%xmm13, %%xmm13\n\t"
                         "mulpd %%xmm14, %%xmm14\n\t"
                         "mulpd %%xmm15, %%xmm15\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                        : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
}

void gflops_double_add(double* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "movupd (%%r8),  %%xmm0\n\t"
                         "movupd (%%r8),  %%xmm1\n\t"
                         "movupd (%%r8),  %%xmm2\n\t"
                         "movupd (%%r8),  %%xmm3\n\t"
                         "movupd (%%r8),  %%xmm4\n\t"
                         "movupd (%%r8),  %%xmm5\n\t"
                         "movupd (%%r8),  %%xmm6\n\t"
                         "movupd (%%r8),  %%xmm7\n\t"
                         "movupd (%%r8),  %%xmm8\n\t"
                         "movupd (%%r8),  %%xmm9\n\t"
                         "movupd (%%r8), %%xmm10\n\t"
                         "movupd (%%r8), %%xmm11\n\t"
                         "movupd (%%r8), %%xmm12\n\t"
                         "movupd (%%r8), %%xmm13\n\t"
                         "movupd (%%r8), %%xmm14\n\t"
                         "movupd (%%r8), %%xmm15\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "addpd  %%xmm0,  %%xmm0\n\t"
                         "addpd  %%xmm1,  %%xmm1\n\t"
                         "addpd  %%xmm2,  %%xmm2\n\t"
                         "addpd  %%xmm3,  %%xmm3\n\t"
                         "addpd  %%xmm4,  %%xmm4\n\t"
                         "addpd  %%xmm5,  %%xmm5\n\t"
                         "addpd  %%xmm6,  %%xmm6\n\t"
                         "addpd  %%xmm7,  %%xmm7\n\t"
                         "addpd  %%xmm8,  %%xmm8\n\t"
                         "addpd  %%xmm9,  %%xmm9\n\t"
                         "addpd %%xmm10, %%xmm10\n\t"
                         "addpd %%xmm11, %%xmm11\n\t"
                         "addpd %%xmm12, %%xmm12\n\t"
                         "addpd %%xmm13, %%xmm13\n\t"
                         "addpd %%xmm14, %%xmm14\n\t"
                         "addpd %%xmm15, %%xmm15\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                        : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
}

#ifdef BENCH_SLMSSE
void gflops_double_madd(double* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "movupd (%%r8),  %%xmm0\n\t"
                         "movupd (%%r8),  %%xmm1\n\t"
                         "movupd (%%r8),  %%xmm2\n\t"
                         "movupd (%%r8),  %%xmm3\n\t"
                         "movupd (%%r8),  %%xmm4\n\t"
                         "movupd (%%r8),  %%xmm5\n\t"
                         "movupd (%%r8),  %%xmm6\n\t"
                         "movupd (%%r8),  %%xmm7\n\t"
                         "movupd (%%r8),  %%xmm8\n\t"
                         "movupd (%%r8),  %%xmm9\n\t"
                         "movupd (%%r8), %%xmm10\n\t"
                         "movupd (%%r8), %%xmm11\n\t"
                         "movupd (%%r8), %%xmm12\n\t"
                         "movupd (%%r8), %%xmm13\n\t"
                         "movupd (%%r8), %%xmm14\n\t"
                         "movupd (%%r8), %%xmm15\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "addpd  %%xmm0,  %%xmm0\n\t"
                         "addpd  %%xmm1,  %%xmm1\n\t"
                         "mulpd  %%xmm2,  %%xmm2\n\t"
                         "addpd  %%xmm3,  %%xmm3\n\t"
                         "addpd  %%xmm4,  %%xmm4\n\t"
                         "mulpd  %%xmm5,  %%xmm5\n\t"
                         "addpd  %%xmm6,  %%xmm6\n\t"
                         "addpd  %%xmm7,  %%xmm7\n\t"
                         "mulpd  %%xmm8,  %%xmm8\n\t"
                         "addpd  %%xmm9,  %%xmm9\n\t"
                         "addpd %%xmm10, %%xmm10\n\t"
                         "mulpd %%xmm11, %%xmm11\n\t"
                         "addpd %%xmm12, %%xmm12\n\t"
                         "addpd %%xmm13, %%xmm13\n\t"
                         "mulpd %%xmm14, %%xmm14\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                        : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
}
#else
void gflops_double_madd(double* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "movupd (%%r8),  %%xmm0\n\t"
                         "movupd (%%r8),  %%xmm1\n\t"
                         "movupd (%%r8),  %%xmm2\n\t"
                         "movupd (%%r8),  %%xmm3\n\t"
                         "movupd (%%r8),  %%xmm4\n\t"
                         "movupd (%%r8),  %%xmm5\n\t"
                         "movupd (%%r8),  %%xmm6\n\t"
                         "movupd (%%r8),  %%xmm7\n\t"
                         "movupd (%%r8),  %%xmm8\n\t"
                         "movupd (%%r8),  %%xmm9\n\t"
                         "movupd (%%r8), %%xmm10\n\t"
                         "movupd (%%r8), %%xmm11\n\t"
                         "movupd (%%r8), %%xmm12\n\t"
                         "movupd (%%r8), %%xmm13\n\t"
                         "movupd (%%r8), %%xmm14\n\t"
                         "movupd (%%r8), %%xmm15\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "mulpd  %%xmm0,  %%xmm0\n\t"
                         "addpd  %%xmm1,  %%xmm1\n\t"
                         "mulpd  %%xmm2,  %%xmm2\n\t"
                         "addpd  %%xmm3,  %%xmm3\n\t"
                         "mulpd  %%xmm4,  %%xmm4\n\t"
                         "addpd  %%xmm5,  %%xmm5\n\t"
                         "mulpd  %%xmm6,  %%xmm6\n\t"
                         "addpd  %%xmm7,  %%xmm7\n\t"
                         "mulpd  %%xmm8,  %%xmm8\n\t"
                         "addpd  %%xmm9,  %%xmm9\n\t"
                         "mulpd %%xmm10, %%xmm10\n\t"
                         "addpd %%xmm11, %%xmm11\n\t"
                         "mulpd %%xmm12, %%xmm12\n\t"
                         "addpd %%xmm13, %%xmm13\n\t"
                         "mulpd %%xmm14, %%xmm14\n\t"
                         "addpd %%xmm15, %%xmm15\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                        : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
}
#endif
