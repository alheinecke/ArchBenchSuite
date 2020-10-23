/******************************************************************************
** Copyright (c) 2020-2020, Intel Corporation                                **
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
/* Sanchit Misra (Intel Corp), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <err.h>
#include <unistd.h>
#include <syscall.h>
#include <linux/perf_event.h>

#include "counters.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct perf_skx_uc_fd{
  int fd_act_rd[SKX_NIMC];
  int fd_act_wr[SKX_NIMC];
  int fd_cas_rd[SKX_NIMC];
  int fd_cas_wr[SKX_NIMC];
  int fd_imc_clockticks[SKX_NIMC];
  int fd_cha_rd[SKX_NCHA];
  int fd_cha_wr[SKX_NCHA];
  int fd_vert_ring_bl_in_use_up[SKX_NCHA];
  int fd_vert_ring_bl_in_use_dn[SKX_NCHA];
  int fd_horz_ring_bl_in_use_lf[SKX_NCHA];
  int fd_horz_ring_bl_in_use_rt[SKX_NCHA];
  int fd_cha_clockticks[SKX_NCHA];
  ctrs_skx_uc_exp exp;
} perf_skx_uc_fd;

static perf_skx_uc_fd gbl_perf_fd;

static int perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags) {
  int ret;
  ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                  group_fd, flags);
  return ret;
}

void evsetup(const char *ename, int *fd, int event, int umask) {
  char fname[1024];
  snprintf(fname, sizeof(fname), "%s/type", ename);
  FILE *fp = fopen(fname, "r");
  if (fp == 0) {
    err(1, "open %s", fname);
  }
  int type;
  int ret = fscanf(fp, "%d", &type);
  assert(ret == 1);
  fclose(fp);
#if 0
  printf("Using PMU type %d from %s\n", type, ename);
#endif

  struct perf_event_attr hw = {};
  hw.size = sizeof(hw);
  hw.type = type;
#if 0
  see /sys/devices/uncore_*/format/*
  Although are events we are using here are configured in the same way, 
  we should read the format.

  hw.read_format = PERF_FORMAT_GROUP;
  unfortunately the below only works within a single PMU; might
  as well just read them one at a time

  on two socket system we would need to create a second set for the 
  second socket
#endif
  hw.config = event | (umask << 8);
  int cpu = 0;
  int pid = -1;
  *fd = perf_event_open(&hw, pid, cpu, -1, 0);
  if (*fd == -1) {
    err(1, "CPU %d, box %s, event 0x%lx", cpu, ename, hw.config);
  }
}

void setup_skx_uc_ctrs( ctrs_skx_uc_exp exp ) {
  int ret;
  char fname[1024];
  int mc, cha;

  for ( mc = 0; mc < SKX_NIMC; ++mc ) {
    snprintf(fname, sizeof(fname), "/sys/devices/uncore_imc_%d",mc);
    if ( exp == CTRS_EXP_DRAM_ACT ) {
      evsetup(fname, &gbl_perf_fd.fd_act_rd[mc], 0x01, 0x01);
      evsetup(fname, &gbl_perf_fd.fd_act_wr[mc], 0x01, 0x02);
      evsetup(fname, &gbl_perf_fd.fd_imc_clockticks[mc], 0x00, 0x00);
    } else if ( exp == CTRS_EXP_DRAM_CAS ) {
      evsetup(fname, &gbl_perf_fd.fd_cas_rd[mc], 0x04, 0x03);
      evsetup(fname, &gbl_perf_fd.fd_cas_wr[mc], 0x04, 0x0C);
      evsetup(fname, &gbl_perf_fd.fd_imc_clockticks[mc], 0x00, 0x00);
    } else {
      /* nothing */
    }
  }

  for ( cha = 0; cha < SKX_NCHA; ++cha ) {
    snprintf(fname, sizeof(fname), "/sys/devices/uncore_cha_%d",cha);
    if ( exp == CTRS_EXP_CHA_ACT ) {
      evsetup(fname, &gbl_perf_fd.fd_cha_rd[cha], 0x50, 0x03);
      evsetup(fname, &gbl_perf_fd.fd_cha_wr[cha], 0x50, 0x0C);
      evsetup(fname, &gbl_perf_fd.fd_cha_clockticks[cha], 0x00, 0x00);
    } else if ( exp == CTRS_EXP_CHA_BL_VERT ) {
      evsetup(fname, &gbl_perf_fd.fd_vert_ring_bl_in_use_up[cha], 0xAA, 0x03);
      evsetup(fname, &gbl_perf_fd.fd_vert_ring_bl_in_use_dn[cha], 0xAA, 0x0C);
      evsetup(fname, &gbl_perf_fd.fd_cha_clockticks[cha], 0x00, 0x00);
    } else if ( exp == CTRS_EXP_CHA_BL_HORZ ) {
      evsetup(fname, &gbl_perf_fd.fd_horz_ring_bl_in_use_lf[cha], 0xAA, 0x03);
      evsetup(fname, &gbl_perf_fd.fd_horz_ring_bl_in_use_rt[cha], 0xAA, 0x0C);
      evsetup(fname, &gbl_perf_fd.fd_cha_clockticks[cha], 0x00, 0x00);
    } else {
      /* nothing */
    }
  }

  gbl_perf_fd.exp = exp;
}

static uint64_t readctr(int fd) {
  uint64_t data;
  size_t s = read(fd, &data, sizeof(data));

  if (s != sizeof(uint64_t)) {
    err(1, "read counter %lu", s);
  }

  return data;
}

void read_skx_uc_ctrs( ctrs_skx_uc *c ) {
  int mc, cha;
  for ( mc = 0; mc < SKX_NIMC; ++mc ) {
    if ( gbl_perf_fd.exp == CTRS_EXP_DRAM_ACT ) {
      c->act_rd[mc] = readctr(gbl_perf_fd.fd_act_rd[mc]);
      c->act_wr[mc] = readctr(gbl_perf_fd.fd_act_wr[mc]);
      c->imc_clockticks[mc] = readctr(gbl_perf_fd.fd_imc_clockticks[mc]);
    } else if ( gbl_perf_fd.exp == CTRS_EXP_DRAM_CAS ) {
      c->cas_rd[mc] = readctr(gbl_perf_fd.fd_cas_rd[mc]);
      c->cas_wr[mc] = readctr(gbl_perf_fd.fd_cas_wr[mc]);
      c->imc_clockticks[mc] = readctr(gbl_perf_fd.fd_imc_clockticks[mc]);
    } else {
      /* nothing */
    }
  }

  for ( cha = 0; cha < SKX_NCHA; ++cha ) {
    if ( gbl_perf_fd.exp == CTRS_EXP_CHA_ACT ) {
      c->cha_rd[cha] = readctr(gbl_perf_fd.fd_cha_rd[cha]);
      c->cha_wr[cha] = readctr(gbl_perf_fd.fd_cha_wr[cha]);
      c->cha_clockticks[cha] = readctr(gbl_perf_fd.fd_cha_clockticks[cha]);
    } else if ( gbl_perf_fd.exp == CTRS_EXP_CHA_BL_VERT ) {
      c->vert_ring_bl_in_use_up[cha] = readctr(gbl_perf_fd.fd_vert_ring_bl_in_use_up[cha]);
      c->vert_ring_bl_in_use_dn[cha] = readctr(gbl_perf_fd.fd_vert_ring_bl_in_use_dn[cha]);
      c->cha_clockticks[cha] = readctr(gbl_perf_fd.fd_cha_clockticks[cha]);
    } else if ( gbl_perf_fd.exp == CTRS_EXP_CHA_BL_HORZ ) {
      c->horz_ring_bl_in_use_lf[cha] = readctr(gbl_perf_fd.fd_horz_ring_bl_in_use_lf[cha]);
      c->horz_ring_bl_in_use_rt[cha] = readctr(gbl_perf_fd.fd_horz_ring_bl_in_use_rt[cha]);
      c->cha_clockticks[cha] = readctr(gbl_perf_fd.fd_cha_clockticks[cha]);
    } else {
      /* nothing */
    }
  }

  c->exp = gbl_perf_fd.exp;
}

void zero_skx_uc_ctrs( ctrs_skx_uc *c ) {
  int mc, cha;
  for ( mc = 0; mc < SKX_NIMC; ++mc ) {
    c->act_rd[mc] = 0;
    c->act_wr[mc] = 0;
    c->cas_rd[mc] = 0;
    c->cas_wr[mc] = 0;
    c->imc_clockticks[mc] = 0;
  }

  for ( cha = 0; cha < SKX_NCHA; ++cha ) {
    c->cha_rd[cha] = 0;
    c->cha_wr[cha] = 0;
    c->vert_ring_bl_in_use_up[cha] = 0;
    c->vert_ring_bl_in_use_dn[cha] = 0;
    c->horz_ring_bl_in_use_lf[cha] = 0;
    c->horz_ring_bl_in_use_rt[cha] = 0;
    c->cha_clockticks[cha] = 0;
  }
}

void divi_skx_uc_ctrs( ctrs_skx_uc *c, uint64_t div ) {
  int mc, cha;
  for ( mc = 0; mc < SKX_NIMC; ++mc ) {
    c->act_rd[mc] /= div;
    c->act_wr[mc] /= div;
    c->cas_rd[mc] /= div;
    c->cas_wr[mc] /= div;
    c->imc_clockticks[mc] /= div;
  }

  for ( cha = 0; cha < SKX_NCHA; ++cha ) {
    c->cha_rd[cha] /= div;
    c->cha_wr[cha] /= div;
    c->vert_ring_bl_in_use_up[cha] /= div;
    c->vert_ring_bl_in_use_dn[cha] /= div;
    c->horz_ring_bl_in_use_lf[cha] /= div;
    c->horz_ring_bl_in_use_rt[cha] /= div;
    c->cha_clockticks[cha] /= div;
  }
}

void difa_skx_uc_ctrs( const ctrs_skx_uc *a, const ctrs_skx_uc *b, ctrs_skx_uc* c ) {
  int mc, cha;

  if ( a->exp != b->exp ) {
    printf("exp type for a and b need to be identical!\n");
    return;
  }

  for ( mc = 0; mc < SKX_NIMC; ++mc ) {
    c->act_rd[mc] += b->act_rd[mc] - a->act_rd[mc];
    c->act_wr[mc] += b->act_wr[mc] - a->act_wr[mc];
    c->cas_rd[mc] += b->cas_rd[mc] - a->cas_rd[mc];
    c->cas_wr[mc] += b->cas_wr[mc] - a->cas_wr[mc];
    c->imc_clockticks[mc] += b->imc_clockticks[mc] - a->imc_clockticks[mc];
  }

  for ( cha = 0; cha < SKX_NCHA; ++cha ) {
    c->cha_rd[cha] += b->cha_rd[cha] - a->cha_rd[cha];
    c->cha_wr[cha] += b->cha_wr[cha] - a->cha_wr[cha];
    c->vert_ring_bl_in_use_up[cha] += b->vert_ring_bl_in_use_up[cha] - a->vert_ring_bl_in_use_up[cha];
    c->vert_ring_bl_in_use_dn[cha] += b->vert_ring_bl_in_use_dn[cha] - a->vert_ring_bl_in_use_dn[cha];
    c->horz_ring_bl_in_use_lf[cha] += b->horz_ring_bl_in_use_lf[cha] - a->horz_ring_bl_in_use_lf[cha];
    c->horz_ring_bl_in_use_rt[cha] += b->horz_ring_bl_in_use_rt[cha] - a->horz_ring_bl_in_use_rt[cha];
    c->cha_clockticks[cha] += b->cha_clockticks[cha] - a->cha_clockticks[cha];
  }

  c->exp = a->exp;
}

void get_cas_ddr_bw_skx( const ctrs_skx_uc *c, const double t, bw_gibs* bw ) {
  uint64_t read_bytes;
  uint64_t write_bytes;
  int mc;
  
  read_bytes  = 0;
  write_bytes = 0;

  if ( c->exp != CTRS_EXP_DRAM_CAS ) { 
    printf("exp type need to be CTRS_EXP_DRAM_CAS!\n");
    bw->rd = 0;
    bw->wr = 0;
    return;

  }
   
  for ( mc = 0; mc < SKX_NIMC; ++mc ) {
    read_bytes  += c->cas_rd[mc]*64;
    write_bytes += c->cas_wr[mc]*64;
  }

  bw->rd = (((double)read_bytes )/t)/(1024.0*1024.0*1024.0);
  bw->wr = (((double)write_bytes)/t)/(1024.0*1024.0*1024.0);
}

#ifdef __cplusplus
}
#endif

