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

#ifdef __AVX2__
void gflops_double_fma(double* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "vmovupd (%%r8),  %%ymm0\n\t"
                         "vmovupd (%%r8),  %%ymm1\n\t"
                         "vmovupd (%%r8),  %%ymm2\n\t"
                         "vmovupd (%%r8),  %%ymm3\n\t"
                         "vmovupd (%%r8),  %%ymm4\n\t"
                         "vmovupd (%%r8),  %%ymm5\n\t"
                         "vmovupd (%%r8),  %%ymm6\n\t"
                         "vmovupd (%%r8),  %%ymm7\n\t"
                         "vmovupd (%%r8),  %%ymm8\n\t"
                         "vmovupd (%%r8),  %%ymm9\n\t"
                         "vmovupd (%%r8), %%ymm10\n\t"
                         "vmovupd (%%r8), %%ymm11\n\t"
                         "vmovupd (%%r8), %%ymm12\n\t"
                         "vmovupd (%%r8), %%ymm13\n\t"
                         "vmovupd (%%r8), %%ymm14\n\t"
                         "vmovupd (%%r8), %%ymm15\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "vfmadd231pd  %%ymm0,  %%ymm0,  %%ymm0\n\t"
                         "vfmadd231pd  %%ymm1,  %%ymm1,  %%ymm1\n\t"
                         "vfmadd231pd  %%ymm2,  %%ymm2,  %%ymm2\n\t"
                         "vfmadd231pd  %%ymm3,  %%ymm3,  %%ymm3\n\t"
                         "vfmadd231pd  %%ymm4,  %%ymm4,  %%ymm4\n\t"
                         "vfmadd231pd  %%ymm5,  %%ymm5,  %%ymm5\n\t"
                         "vfmadd231pd  %%ymm6,  %%ymm6,  %%ymm6\n\t"
                         "vfmadd231pd  %%ymm7,  %%ymm7,  %%ymm7\n\t"
                         "vfmadd231pd  %%ymm8,  %%ymm8,  %%ymm8\n\t"
                         "vfmadd231pd  %%ymm9,  %%ymm9,  %%ymm9\n\t"
                         "vfmadd231pd %%ymm10, %%ymm10, %%ymm10\n\t"
                         "vfmadd231pd %%ymm11, %%ymm11, %%ymm11\n\t"
                         "vfmadd231pd %%ymm12, %%ymm12, %%ymm12\n\t"
                         "vfmadd231pd %%ymm13, %%ymm13, %%ymm13\n\t"
                         "vfmadd231pd %%ymm14, %%ymm14, %%ymm14\n\t"
                         "vfmadd231pd %%ymm15, %%ymm15, %%ymm15\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                        : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
}
#else
void gflops_double_fma(double* data) {
  std::cout << "FMA is not available on this architecture" << std::endl;
}
#endif

void gflops_double_mul(double* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "vmovupd (%%r8),  %%ymm0\n\t"
                         "vmovupd (%%r8),  %%ymm1\n\t"
                         "vmovupd (%%r8),  %%ymm2\n\t"
                         "vmovupd (%%r8),  %%ymm3\n\t"
                         "vmovupd (%%r8),  %%ymm4\n\t"
                         "vmovupd (%%r8),  %%ymm5\n\t"
                         "vmovupd (%%r8),  %%ymm6\n\t"
                         "vmovupd (%%r8),  %%ymm7\n\t"
                         "vmovupd (%%r8),  %%ymm8\n\t"
                         "vmovupd (%%r8),  %%ymm9\n\t"
                         "vmovupd (%%r8), %%ymm10\n\t"
                         "vmovupd (%%r8), %%ymm11\n\t"
                         "vmovupd (%%r8), %%ymm12\n\t"
                         "vmovupd (%%r8), %%ymm13\n\t"
                         "vmovupd (%%r8), %%ymm14\n\t"
                         "vmovupd (%%r8), %%ymm15\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "vmulpd  %%ymm0,  %%ymm0,  %%ymm0\n\t"
                         "vmulpd  %%ymm1,  %%ymm1,  %%ymm1\n\t"
                         "vmulpd  %%ymm2,  %%ymm2,  %%ymm2\n\t"
                         "vmulpd  %%ymm3,  %%ymm3,  %%ymm3\n\t"
                         "vmulpd  %%ymm4,  %%ymm4,  %%ymm4\n\t"
                         "vmulpd  %%ymm5,  %%ymm5,  %%ymm5\n\t"
                         "vmulpd  %%ymm6,  %%ymm6,  %%ymm6\n\t"
                         "vmulpd  %%ymm7,  %%ymm7,  %%ymm7\n\t"
                         "vmulpd  %%ymm8,  %%ymm8,  %%ymm8\n\t"
                         "vmulpd  %%ymm9,  %%ymm9,  %%ymm9\n\t"
                         "vmulpd %%ymm10, %%ymm10, %%ymm10\n\t"
                         "vmulpd %%ymm11, %%ymm11, %%ymm11\n\t"
                         "vmulpd %%ymm12, %%ymm12, %%ymm12\n\t"
                         "vmulpd %%ymm13, %%ymm13, %%ymm13\n\t"
                         "vmulpd %%ymm14, %%ymm14, %%ymm14\n\t"
                         "vmulpd %%ymm15, %%ymm15, %%ymm15\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                        : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
}

void gflops_double_add(double* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "vmovupd (%%r8),  %%ymm0\n\t"
                         "vmovupd (%%r8),  %%ymm1\n\t"
                         "vmovupd (%%r8),  %%ymm2\n\t"
                         "vmovupd (%%r8),  %%ymm3\n\t"
                         "vmovupd (%%r8),  %%ymm4\n\t"
                         "vmovupd (%%r8),  %%ymm5\n\t"
                         "vmovupd (%%r8),  %%ymm6\n\t"
                         "vmovupd (%%r8),  %%ymm7\n\t"
                         "vmovupd (%%r8),  %%ymm8\n\t"
                         "vmovupd (%%r8),  %%ymm9\n\t"
                         "vmovupd (%%r8), %%ymm10\n\t"
                         "vmovupd (%%r8), %%ymm11\n\t"
                         "vmovupd (%%r8), %%ymm12\n\t"
                         "vmovupd (%%r8), %%ymm13\n\t"
                         "vmovupd (%%r8), %%ymm14\n\t"
                         "vmovupd (%%r8), %%ymm15\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "vaddpd  %%ymm0,  %%ymm0,  %%ymm0\n\t"
                         "vaddpd  %%ymm1,  %%ymm1,  %%ymm1\n\t"
                         "vaddpd  %%ymm2,  %%ymm2,  %%ymm2\n\t"
                         "vaddpd  %%ymm3,  %%ymm3,  %%ymm3\n\t"
                         "vaddpd  %%ymm4,  %%ymm4,  %%ymm4\n\t"
                         "vaddpd  %%ymm5,  %%ymm5,  %%ymm5\n\t"
                         "vaddpd  %%ymm6,  %%ymm6,  %%ymm6\n\t"
                         "vaddpd  %%ymm7,  %%ymm7,  %%ymm7\n\t"
                         "vaddpd  %%ymm8,  %%ymm8,  %%ymm8\n\t"
                         "vaddpd  %%ymm9,  %%ymm9,  %%ymm9\n\t"
                         "vaddpd %%ymm10, %%ymm10, %%ymm10\n\t"
                         "vaddpd %%ymm11, %%ymm11, %%ymm11\n\t"
                         "vaddpd %%ymm12, %%ymm12, %%ymm12\n\t"
                         "vaddpd %%ymm13, %%ymm13, %%ymm13\n\t"
                         "vaddpd %%ymm14, %%ymm14, %%ymm14\n\t"
                         "vaddpd %%ymm15, %%ymm15, %%ymm15\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                        : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
}

void gflops_double_madd(double* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "vmovupd (%%r8),  %%ymm0\n\t"
                         "vmovupd (%%r8),  %%ymm1\n\t"
                         "vmovupd (%%r8),  %%ymm2\n\t"
                         "vmovupd (%%r8),  %%ymm3\n\t"
                         "vmovupd (%%r8),  %%ymm4\n\t"
                         "vmovupd (%%r8),  %%ymm5\n\t"
                         "vmovupd (%%r8),  %%ymm6\n\t"
                         "vmovupd (%%r8),  %%ymm7\n\t"
                         "vmovupd (%%r8),  %%ymm8\n\t"
                         "vmovupd (%%r8),  %%ymm9\n\t"
                         "vmovupd (%%r8), %%ymm10\n\t"
                         "vmovupd (%%r8), %%ymm11\n\t"
                         "vmovupd (%%r8), %%ymm12\n\t"
                         "vmovupd (%%r8), %%ymm13\n\t"
                         "vmovupd (%%r8), %%ymm14\n\t"
                         "vmovupd (%%r8), %%ymm15\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "vmulpd  %%ymm0,  %%ymm0,  %%ymm0\n\t"
                         "vaddpd  %%ymm1,  %%ymm1,  %%ymm1\n\t"
                         "vmulpd  %%ymm2,  %%ymm2,  %%ymm2\n\t"
                         "vaddpd  %%ymm3,  %%ymm3,  %%ymm3\n\t"
                         "vmulpd  %%ymm4,  %%ymm4,  %%ymm4\n\t"
                         "vaddpd  %%ymm5,  %%ymm5,  %%ymm5\n\t"
                         "vmulpd  %%ymm6,  %%ymm6,  %%ymm6\n\t"
                         "vaddpd  %%ymm7,  %%ymm7,  %%ymm7\n\t"
                         "vmulpd  %%ymm8,  %%ymm8,  %%ymm8\n\t"
                         "vaddpd  %%ymm9,  %%ymm9,  %%ymm9\n\t"
                         "vmulpd %%ymm10, %%ymm10, %%ymm10\n\t"
                         "vaddpd %%ymm11, %%ymm11, %%ymm11\n\t"
                         "vmulpd %%ymm12, %%ymm12, %%ymm12\n\t"
                         "vaddpd %%ymm13, %%ymm13, %%ymm13\n\t"
                         "vmulpd %%ymm14, %%ymm14, %%ymm14\n\t"
                         "vaddpd %%ymm15, %%ymm15, %%ymm15\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                        : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
}

