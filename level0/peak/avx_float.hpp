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

void gflops_float_qfma(float* data) {
  std::cout << "QFMA is not available on this architecture" << std::endl;
}

#ifdef __AVX2__
void gflops_float_fma(float* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "vmovups (%%r8),  %%ymm0\n\t"
                         "vmovups (%%r8),  %%ymm1\n\t"
                         "vmovups (%%r8),  %%ymm2\n\t"
                         "vmovups (%%r8),  %%ymm3\n\t"
                         "vmovups (%%r8),  %%ymm4\n\t"
                         "vmovups (%%r8),  %%ymm5\n\t"
                         "vmovups (%%r8),  %%ymm6\n\t"
                         "vmovups (%%r8),  %%ymm7\n\t"
                         "vmovups (%%r8),  %%ymm8\n\t"
                         "vmovups (%%r8),  %%ymm9\n\t"
                         "vmovups (%%r8), %%ymm10\n\t"
                         "vmovups (%%r8), %%ymm11\n\t"
                         "vmovups (%%r8), %%ymm12\n\t"
                         "vmovups (%%r8), %%ymm13\n\t"
                         "vmovups (%%r8), %%ymm14\n\t"
                         "vmovups (%%r8), %%ymm15\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "vfmadd231ps  %%ymm0,  %%ymm0,  %%ymm0\n\t"
                         "vfmadd231ps  %%ymm1,  %%ymm1,  %%ymm1\n\t"
                         "vfmadd231ps  %%ymm2,  %%ymm2,  %%ymm2\n\t"
                         "vfmadd231ps  %%ymm3,  %%ymm3,  %%ymm3\n\t"
                         "vfmadd231ps  %%ymm4,  %%ymm4,  %%ymm4\n\t"
                         "vfmadd231ps  %%ymm5,  %%ymm5,  %%ymm5\n\t"
                         "vfmadd231ps  %%ymm6,  %%ymm6,  %%ymm6\n\t"
                         "vfmadd231ps  %%ymm7,  %%ymm7,  %%ymm7\n\t"
                         "vfmadd231ps  %%ymm8,  %%ymm8,  %%ymm8\n\t"
                         "vfmadd231ps  %%ymm9,  %%ymm9,  %%ymm9\n\t"
                         "vfmadd231ps %%ymm10, %%ymm10, %%ymm10\n\t"
                         "vfmadd231ps %%ymm11, %%ymm11, %%ymm11\n\t"
                         "vfmadd231ps %%ymm12, %%ymm12, %%ymm12\n\t"
                         "vfmadd231ps %%ymm13, %%ymm13, %%ymm13\n\t"
                         "vfmadd231ps %%ymm14, %%ymm14, %%ymm14\n\t"
                         "vfmadd231ps %%ymm15, %%ymm15, %%ymm15\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                        : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
}
#else
void gflops_float_fma(float* data) {
  std::cout << "FMA is not available on this architecture" << std::endl;
}
#endif

void gflops_float_mul(float* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "vmovups (%%r8),  %%ymm0\n\t"
                         "vmovups (%%r8),  %%ymm1\n\t"
                         "vmovups (%%r8),  %%ymm2\n\t"
                         "vmovups (%%r8),  %%ymm3\n\t"
                         "vmovups (%%r8),  %%ymm4\n\t"
                         "vmovups (%%r8),  %%ymm5\n\t"
                         "vmovups (%%r8),  %%ymm6\n\t"
                         "vmovups (%%r8),  %%ymm7\n\t"
                         "vmovups (%%r8),  %%ymm8\n\t"
                         "vmovups (%%r8),  %%ymm9\n\t"
                         "vmovups (%%r8), %%ymm10\n\t"
                         "vmovups (%%r8), %%ymm11\n\t"
                         "vmovups (%%r8), %%ymm12\n\t"
                         "vmovups (%%r8), %%ymm13\n\t"
                         "vmovups (%%r8), %%ymm14\n\t"
                         "vmovups (%%r8), %%ymm15\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "vmulps  %%ymm0,  %%ymm0,  %%ymm0\n\t"
                         "vmulps  %%ymm1,  %%ymm1,  %%ymm1\n\t"
                         "vmulps  %%ymm2,  %%ymm2,  %%ymm2\n\t"
                         "vmulps  %%ymm3,  %%ymm3,  %%ymm3\n\t"
                         "vmulps  %%ymm4,  %%ymm4,  %%ymm4\n\t"
                         "vmulps  %%ymm5,  %%ymm5,  %%ymm5\n\t"
                         "vmulps  %%ymm6,  %%ymm6,  %%ymm6\n\t"
                         "vmulps  %%ymm7,  %%ymm7,  %%ymm7\n\t"
                         "vmulps  %%ymm8,  %%ymm8,  %%ymm8\n\t"
                         "vmulps  %%ymm9,  %%ymm9,  %%ymm9\n\t"
                         "vmulps %%ymm10, %%ymm10, %%ymm10\n\t"
                         "vmulps %%ymm11, %%ymm11, %%ymm11\n\t"
                         "vmulps %%ymm12, %%ymm12, %%ymm12\n\t"
                         "vmulps %%ymm13, %%ymm13, %%ymm13\n\t"
                         "vmulps %%ymm14, %%ymm14, %%ymm14\n\t"
                         "vmulps %%ymm15, %%ymm15, %%ymm15\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                        : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
}

void gflops_float_add(float* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "vmovups (%%r8),  %%ymm0\n\t"
                         "vmovups (%%r8),  %%ymm1\n\t"
                         "vmovups (%%r8),  %%ymm2\n\t"
                         "vmovups (%%r8),  %%ymm3\n\t"
                         "vmovups (%%r8),  %%ymm4\n\t"
                         "vmovups (%%r8),  %%ymm5\n\t"
                         "vmovups (%%r8),  %%ymm6\n\t"
                         "vmovups (%%r8),  %%ymm7\n\t"
                         "vmovups (%%r8),  %%ymm8\n\t"
                         "vmovups (%%r8),  %%ymm9\n\t"
                         "vmovups (%%r8), %%ymm10\n\t"
                         "vmovups (%%r8), %%ymm11\n\t"
                         "vmovups (%%r8), %%ymm12\n\t"
                         "vmovups (%%r8), %%ymm13\n\t"
                         "vmovups (%%r8), %%ymm14\n\t"
                         "vmovups (%%r8), %%ymm15\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "vaddps  %%ymm0,  %%ymm0,  %%ymm0\n\t"
                         "vaddps  %%ymm1,  %%ymm1,  %%ymm1\n\t"
                         "vaddps  %%ymm2,  %%ymm2,  %%ymm2\n\t"
                         "vaddps  %%ymm3,  %%ymm3,  %%ymm3\n\t"
                         "vaddps  %%ymm4,  %%ymm4,  %%ymm4\n\t"
                         "vaddps  %%ymm5,  %%ymm5,  %%ymm5\n\t"
                         "vaddps  %%ymm6,  %%ymm6,  %%ymm6\n\t"
                         "vaddps  %%ymm7,  %%ymm7,  %%ymm7\n\t"
                         "vaddps  %%ymm8,  %%ymm8,  %%ymm8\n\t"
                         "vaddps  %%ymm9,  %%ymm9,  %%ymm9\n\t"
                         "vaddps %%ymm10, %%ymm10, %%ymm10\n\t"
                         "vaddps %%ymm11, %%ymm11, %%ymm11\n\t"
                         "vaddps %%ymm12, %%ymm12, %%ymm12\n\t"
                         "vaddps %%ymm13, %%ymm13, %%ymm13\n\t"
                         "vaddps %%ymm14, %%ymm14, %%ymm14\n\t"
                         "vaddps %%ymm15, %%ymm15, %%ymm15\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                        : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
}

void gflops_float_madd(float* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "vmovups (%%r8),  %%ymm0\n\t"
                         "vmovups (%%r8),  %%ymm1\n\t"
                         "vmovups (%%r8),  %%ymm2\n\t"
                         "vmovups (%%r8),  %%ymm3\n\t"
                         "vmovups (%%r8),  %%ymm4\n\t"
                         "vmovups (%%r8),  %%ymm5\n\t"
                         "vmovups (%%r8),  %%ymm6\n\t"
                         "vmovups (%%r8),  %%ymm7\n\t"
                         "vmovups (%%r8),  %%ymm8\n\t"
                         "vmovups (%%r8),  %%ymm9\n\t"
                         "vmovups (%%r8), %%ymm10\n\t"
                         "vmovups (%%r8), %%ymm11\n\t"
                         "vmovups (%%r8), %%ymm12\n\t"
                         "vmovups (%%r8), %%ymm13\n\t"
                         "vmovups (%%r8), %%ymm14\n\t"
                         "vmovups (%%r8), %%ymm15\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "vmulps  %%ymm0,  %%ymm0,  %%ymm0\n\t"
                         "vaddps  %%ymm1,  %%ymm1,  %%ymm1\n\t"
                         "vmulps  %%ymm2,  %%ymm2,  %%ymm2\n\t"
                         "vaddps  %%ymm3,  %%ymm3,  %%ymm3\n\t"
                         "vmulps  %%ymm4,  %%ymm4,  %%ymm4\n\t"
                         "vaddps  %%ymm5,  %%ymm5,  %%ymm5\n\t"
                         "vmulps  %%ymm6,  %%ymm6,  %%ymm6\n\t"
                         "vaddps  %%ymm7,  %%ymm7,  %%ymm7\n\t"
                         "vmulps  %%ymm8,  %%ymm8,  %%ymm8\n\t"
                         "vaddps  %%ymm9,  %%ymm9,  %%ymm9\n\t"
                         "vmulps %%ymm10, %%ymm10, %%ymm10\n\t"
                         "vaddps %%ymm11, %%ymm11, %%ymm11\n\t"
                         "vmulps %%ymm12, %%ymm12, %%ymm12\n\t"
                         "vaddps %%ymm13, %%ymm13, %%ymm13\n\t"
                         "vmulps %%ymm14, %%ymm14, %%ymm14\n\t"
                         "vaddps %%ymm15, %%ymm15, %%ymm15\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                        : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
}

