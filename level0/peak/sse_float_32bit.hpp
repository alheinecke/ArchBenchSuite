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
  std::cout << "F64 QFMA is not available on this architecture" << std::endl;
}

void gflops_double_fma(double* data) {
  std::cout << "F64 FMA is not available on this architecture" << std::endl;
}

void gflops_double_mul(double* data) {
  std::cout << "F64 MUL is not available on this architecture" << std::endl;
}

void gflops_double_add(double* data) {
  std::cout << "F64 ADD is not available on this architecture" << std::endl;
}

void gflops_double_madd(double* data) {
  std::cout << "F64 MULADD is not available on this architecture" << std::endl;
}

void gflops_float_qfma(float* data) {
  std::cout << "F32 QFMA is not available on this architecture" << std::endl;
}

void gflops_float_fma(float* data) {
  std::cout << "F32 FMA is not available on this architecture" << std::endl;
}

void gflops_float_mul(float* data) {
    __asm__ __volatile__("movl %0, %%eax\n\t"
                         "movups (%%eax),  %%xmm0\n\t"
                         "movups (%%eax),  %%xmm1\n\t"
                         "movups (%%eax),  %%xmm2\n\t"
                         "movups (%%eax),  %%xmm3\n\t"
                         "movups (%%eax),  %%xmm4\n\t"
                         "movups (%%eax),  %%xmm5\n\t"
                         "movups (%%eax),  %%xmm6\n\t"
                         "movups (%%eax),  %%xmm7\n\t"
                         "movl $100000000, %%ebx\n\t" 
                         "1:\n\t"
                         "subl $1, %%ebx\n\t"
                         "mulps  %%xmm0,  %%xmm0\n\t"
                         "mulps  %%xmm1,  %%xmm1\n\t"
                         "mulps  %%xmm2,  %%xmm2\n\t"
                         "mulps  %%xmm3,  %%xmm3\n\t"
                         "mulps  %%xmm4,  %%xmm4\n\t"
                         "mulps  %%xmm5,  %%xmm5\n\t"
                         "mulps  %%xmm6,  %%xmm6\n\t"
                         "mulps  %%xmm7,  %%xmm7\n\t"
                         "cmpl $0, %%ebx\n\t"
                         "jg 1b\n\t"
                         : : "m"(data) : "eax","ebx","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7");
}

void gflops_float_add(float* data) {
    __asm__ __volatile__("movl %0, %%eax\n\t"
                         "movups (%%eax),  %%xmm0\n\t"
                         "movups (%%eax),  %%xmm1\n\t"
                         "movups (%%eax),  %%xmm2\n\t"
                         "movups (%%eax),  %%xmm3\n\t"
                         "movups (%%eax),  %%xmm4\n\t"
                         "movups (%%eax),  %%xmm5\n\t"
                         "movups (%%eax),  %%xmm6\n\t"
                         "movups (%%eax),  %%xmm7\n\t"
                         "movl $100000000, %%ebx\n\t" 
                         "1:\n\t"
                         "nop\n\t"
                         "nop\n\t"
                         "subl $1, %%ebx\n\t"
                         "addps  %%xmm0,  %%xmm0\n\t"
                         "addps  %%xmm1,  %%xmm1\n\t"
                         "addps  %%xmm2,  %%xmm2\n\t"
                         "addps  %%xmm3,  %%xmm3\n\t"
                         "addps  %%xmm4,  %%xmm4\n\t"
                         "addps  %%xmm5,  %%xmm5\n\t"
                         "addps  %%xmm6,  %%xmm6\n\t"
                         "addps  %%xmm7,  %%xmm7\n\t"
                         "cmpl $0, %%ebx\n\t"
                         "jg 1b\n\t"
                         : : "m"(data) : "eax","ebx","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7");
}

void gflops_float_madd(float* data) {
    __asm__ __volatile__("movl %0, %%eax\n\t"
                         "movups (%%eax),  %%xmm0\n\t"
                         "movups (%%eax),  %%xmm1\n\t"
                         "movups (%%eax),  %%xmm2\n\t"
                         "movups (%%eax),  %%xmm3\n\t"
                         "movups (%%eax),  %%xmm4\n\t"
                         "movups (%%eax),  %%xmm5\n\t"
                         "movups (%%eax),  %%xmm6\n\t"
                         "movups (%%eax),  %%xmm7\n\t"
                         "movl $100000000, %%ebx\n\t" 
                         "1:\n\t"
                         "subl $2, %%ebx\n\t"
                         "mulps  %%xmm0, %%xmm0\n\t"
                         "addps  %%xmm0, %%xmm0\n\t"
                         "mulps  %%xmm1, %%xmm1\n\t"
                         "addps  %%xmm1, %%xmm1\n\t"
                         "mulps  %%xmm2, %%xmm2\n\t"
                         "addps  %%xmm2, %%xmm2\n\t"
                         "mulps  %%xmm3, %%xmm3\n\t"
                         "addps  %%xmm3, %%xmm3\n\t"
                         "mulps  %%xmm4, %%xmm4\n\t"
                         "addps  %%xmm4, %%xmm4\n\t"
                         "mulps  %%xmm5, %%xmm5\n\t"
                         "addps  %%xmm5, %%xmm5\n\t"
                         "mulps  %%xmm6, %%xmm6\n\t"
                         "addps  %%xmm6, %%xmm6\n\t"
                         "mulps  %%xmm7, %%xmm7\n\t"
                         "addps  %%xmm7, %%xmm7\n\t"
                         "cmpl $0, %%ebx\n\t"
                         "jg 1b\n\t"
                         : : "m"(data) : "eax","ebx","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7");
}
