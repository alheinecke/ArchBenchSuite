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

void gflops_int_mul(int* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "vmovupd (%%r8),  %%zmm0\n\t"
                         "vmovupd (%%r8),  %%zmm1\n\t"
                         "vmovupd (%%r8),  %%zmm2\n\t"
                         "vmovupd (%%r8),  %%zmm3\n\t"
                         "vmovupd (%%r8),  %%zmm4\n\t"
                         "vmovupd (%%r8),  %%zmm5\n\t"
                         "vmovupd (%%r8),  %%zmm6\n\t"
                         "vmovupd (%%r8),  %%zmm7\n\t"
                         "vmovupd (%%r8),  %%zmm8\n\t"
                         "vmovupd (%%r8),  %%zmm9\n\t"
                         "vmovupd (%%r8), %%zmm10\n\t"
                         "vmovupd (%%r8), %%zmm11\n\t"
                         "vmovupd (%%r8), %%zmm12\n\t"
                         "vmovupd (%%r8), %%zmm13\n\t"
                         "vmovupd (%%r8), %%zmm14\n\t"
                         "vmovupd (%%r8), %%zmm15\n\t"
                         "vmovupd (%%r8), %%zmm16\n\t"
                         "vmovupd (%%r8), %%zmm17\n\t"
                         "vmovupd (%%r8), %%zmm18\n\t"
                         "vmovupd (%%r8), %%zmm19\n\t"
                         "vmovupd (%%r8), %%zmm20\n\t"
                         "vmovupd (%%r8), %%zmm21\n\t"
                         "vmovupd (%%r8), %%zmm22\n\t"
                         "vmovupd (%%r8), %%zmm23\n\t"
                         "vmovupd (%%r8), %%zmm24\n\t"
                         "vmovupd (%%r8), %%zmm25\n\t"
                         "vmovupd (%%r8), %%zmm26\n\t"
                         "vmovupd (%%r8), %%zmm27\n\t"
                         "vmovupd (%%r8), %%zmm28\n\t"
                         "vmovupd (%%r8), %%zmm29\n\t"
                         "vmovupd (%%r8), %%zmm30\n\t"
                         "vmovupd (%%r8), %%zmm31\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "vpmaddwd  %%zmm0,  %%zmm0,  %%zmm0\n\t"
                         "vpmaddwd  %%zmm1,  %%zmm1,  %%zmm1\n\t"
                         "vpmaddwd  %%zmm2,  %%zmm2,  %%zmm2\n\t"
                         "vpmaddwd  %%zmm3,  %%zmm3,  %%zmm3\n\t"
                         "vpmaddwd  %%zmm4,  %%zmm4,  %%zmm4\n\t"
                         "vpmaddwd  %%zmm5,  %%zmm5,  %%zmm5\n\t"
                         "vpmaddwd  %%zmm6,  %%zmm6,  %%zmm6\n\t"
                         "vpmaddwd  %%zmm7,  %%zmm7,  %%zmm7\n\t"
                         "vpmaddwd  %%zmm8,  %%zmm8,  %%zmm8\n\t"
                         "vpmaddwd  %%zmm9,  %%zmm9,  %%zmm9\n\t"
                         "vpmaddwd %%zmm10, %%zmm10, %%zmm10\n\t"
                         "vpmaddwd %%zmm11, %%zmm11, %%zmm11\n\t"
                         "vpmaddwd %%zmm12, %%zmm12, %%zmm12\n\t"
                         "vpmaddwd %%zmm13, %%zmm13, %%zmm13\n\t"
                         "vpmaddwd %%zmm14, %%zmm14, %%zmm14\n\t"
                         "vpmaddwd %%zmm15, %%zmm15, %%zmm15\n\t"
                         "vpmaddwd %%zmm16, %%zmm16, %%zmm16\n\t"
                         "vpmaddwd %%zmm17, %%zmm17, %%zmm17\n\t"
                         "vpmaddwd %%zmm18, %%zmm18, %%zmm18\n\t"
                         "vpmaddwd %%zmm19, %%zmm19, %%zmm19\n\t"
                         "vpmaddwd %%zmm20, %%zmm20, %%zmm20\n\t"
                         "vpmaddwd %%zmm21, %%zmm21, %%zmm21\n\t"
                         "vpmaddwd %%zmm22, %%zmm22, %%zmm22\n\t"
                         "vpmaddwd %%zmm23, %%zmm23, %%zmm23\n\t"
                         "vpmaddwd %%zmm24, %%zmm24, %%zmm24\n\t"
                         "vpmaddwd %%zmm25, %%zmm25, %%zmm25\n\t"
                         "vpmaddwd %%zmm26, %%zmm26, %%zmm26\n\t"
                         "vpmaddwd %%zmm27, %%zmm27, %%zmm27\n\t"
                         "vpmaddwd %%zmm28, %%zmm28, %%zmm28\n\t"
                         "vpmaddwd %%zmm29, %%zmm29, %%zmm29\n\t"
                         "vpmaddwd %%zmm30, %%zmm30, %%zmm30\n\t"
                         "vpmaddwd %%zmm31, %%zmm31, %%zmm31\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                         : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","xmm16","xmm17","xmm18","xmm19","xmm20","xmm21","xmm22","xmm23","xmm24","xmm25","xmm26","xmm27","xmm28","xmm29","xmm30","xmm31");
}

void gflops_int_add(int* data) {
    __asm__ __volatile__("movq %0, %%r8\n\t"
                         "vmovupd (%%r8),  %%zmm0\n\t"
                         "vmovupd (%%r8),  %%zmm1\n\t"
                         "vmovupd (%%r8),  %%zmm2\n\t"
                         "vmovupd (%%r8),  %%zmm3\n\t"
                         "vmovupd (%%r8),  %%zmm4\n\t"
                         "vmovupd (%%r8),  %%zmm5\n\t"
                         "vmovupd (%%r8),  %%zmm6\n\t"
                         "vmovupd (%%r8),  %%zmm7\n\t"
                         "vmovupd (%%r8),  %%zmm8\n\t"
                         "vmovupd (%%r8),  %%zmm9\n\t"
                         "vmovupd (%%r8), %%zmm10\n\t"
                         "vmovupd (%%r8), %%zmm11\n\t"
                         "vmovupd (%%r8), %%zmm12\n\t"
                         "vmovupd (%%r8), %%zmm13\n\t"
                         "vmovupd (%%r8), %%zmm14\n\t"
                         "vmovupd (%%r8), %%zmm15\n\t"
                         "vmovupd (%%r8), %%zmm16\n\t"
                         "vmovupd (%%r8), %%zmm17\n\t"
                         "vmovupd (%%r8), %%zmm18\n\t"
                         "vmovupd (%%r8), %%zmm19\n\t"
                         "vmovupd (%%r8), %%zmm20\n\t"
                         "vmovupd (%%r8), %%zmm21\n\t"
                         "vmovupd (%%r8), %%zmm22\n\t"
                         "vmovupd (%%r8), %%zmm23\n\t"
                         "vmovupd (%%r8), %%zmm24\n\t"
                         "vmovupd (%%r8), %%zmm25\n\t"
                         "vmovupd (%%r8), %%zmm26\n\t"
                         "vmovupd (%%r8), %%zmm27\n\t"
                         "vmovupd (%%r8), %%zmm28\n\t"
                         "vmovupd (%%r8), %%zmm29\n\t"
                         "vmovupd (%%r8), %%zmm30\n\t"
                         "vmovupd (%%r8), %%zmm31\n\t"
                         "movq $100000000, %%r9\n\t" 
                         "1:\n\t"
                         "subq $1, %%r9\n\t"
                         "vpaddd  %%zmm0,  %%zmm0,  %%zmm0\n\t"
                         "vpaddd  %%zmm1,  %%zmm1,  %%zmm1\n\t"
                         "vpaddd  %%zmm2,  %%zmm2,  %%zmm2\n\t"
                         "vpaddd  %%zmm3,  %%zmm3,  %%zmm3\n\t"
                         "vpaddd  %%zmm4,  %%zmm4,  %%zmm4\n\t"
                         "vpaddd  %%zmm5,  %%zmm5,  %%zmm5\n\t"
                         "vpaddd  %%zmm6,  %%zmm6,  %%zmm6\n\t"
                         "vpaddd  %%zmm7,  %%zmm7,  %%zmm7\n\t"
                         "vpaddd  %%zmm8,  %%zmm8,  %%zmm8\n\t"
                         "vpaddd  %%zmm9,  %%zmm9,  %%zmm9\n\t"
                         "vpaddd %%zmm10, %%zmm10, %%zmm10\n\t"
                         "vpaddd %%zmm11, %%zmm11, %%zmm11\n\t"
                         "vpaddd %%zmm12, %%zmm12, %%zmm12\n\t"
                         "vpaddd %%zmm13, %%zmm13, %%zmm13\n\t"
                         "vpaddd %%zmm14, %%zmm14, %%zmm14\n\t"
                         "vpaddd %%zmm15, %%zmm15, %%zmm15\n\t"
                         "vpaddd %%zmm16, %%zmm16, %%zmm16\n\t"
                         "vpaddd %%zmm17, %%zmm17, %%zmm17\n\t"
                         "vpaddd %%zmm18, %%zmm18, %%zmm18\n\t"
                         "vpaddd %%zmm19, %%zmm19, %%zmm19\n\t"
                         "vpaddd %%zmm20, %%zmm20, %%zmm20\n\t"
                         "vpaddd %%zmm21, %%zmm21, %%zmm21\n\t"
                         "vpaddd %%zmm22, %%zmm22, %%zmm22\n\t"
                         "vpaddd %%zmm23, %%zmm23, %%zmm23\n\t"
                         "vpaddd %%zmm24, %%zmm24, %%zmm24\n\t"
                         "vpaddd %%zmm25, %%zmm25, %%zmm25\n\t"
                         "vpaddd %%zmm26, %%zmm26, %%zmm26\n\t"
                         "vpaddd %%zmm27, %%zmm27, %%zmm27\n\t"
                         "vpaddd %%zmm28, %%zmm28, %%zmm28\n\t"
                         "vpaddd %%zmm29, %%zmm29, %%zmm29\n\t"
                         "vpaddd %%zmm30, %%zmm30, %%zmm30\n\t"
                         "vpaddd %%zmm31, %%zmm31, %%zmm31\n\t"
                         "cmpq $0, %%r9\n\t"
                         "jg 1b\n\t"
                         : : "m"(data) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","xmm16","xmm17","xmm18","xmm19","xmm20","xmm21","xmm22","xmm23","xmm24","xmm25","xmm26","xmm27","xmm28","xmm29","xmm30","xmm31");
}
