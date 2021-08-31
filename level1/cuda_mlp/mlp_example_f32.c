/******************************************************************************
** Copyright (c) 2013-2021, Alexander Heinecke                               **
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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

void my_init_buf(float* buf, size_t size, int initPos, int initOne) {
  int i;
  memset(buf, 0, sizeof(float)*size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? drand48() : (0.05 - drand48()/10.0)));
  }
}

inline double sec(struct timeval start, struct timeval end) {
  return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec))))/1.0e6;
}

typedef enum my_eltwise_fuse {
  MY_ELTWISE_FUSE_NONE = 0,
  MY_ELTWISE_FUSE_BIAS = 1,
  MY_ELTWISE_FUSE_RELU = 2,
  MY_ELTWISE_FUSE_BIAS_RELU = MY_ELTWISE_FUSE_BIAS | MY_ELTWISE_FUSE_RELU
} my_eltwise_fuse;

typedef enum my_pass {
  MY_PASS_FWD   = 1,
  MY_PASS_BWD_D = 2,
  MY_PASS_BWD_W = 4,
  MY_PASS_BWD   = 6
} my_pass;

typedef struct my_opt_config {
  unsigned int C;
  unsigned int K;
  unsigned int bc;
  unsigned int bk;
  float        lr;
} my_opt_config;

typedef struct my_fc_fwd_config {
  unsigned int N;
  unsigned int C;
  unsigned int K;
  unsigned int bn;
  unsigned int bc;
  unsigned int bk;
  my_eltwise_fuse fuse_type;
  unsigned int fwd_bf;
  unsigned int fwd_2d_blocking;
  unsigned int fwd_col_teams;
  unsigned int fwd_row_teams;
  float alpha;
  float beta;
  cublasHandle_t cuda_handle;
} my_fc_fwd_config;

typedef struct my_fc_bwd_config {
  unsigned int N;
  unsigned int C;
  unsigned int K;
  unsigned int bn;
  unsigned int bc;
  unsigned int bk;
  my_eltwise_fuse fuse_type;
  unsigned int bwd_bf;
  unsigned int bwd_2d_blocking;
  unsigned int bwd_col_teams;
  unsigned int bwd_row_teams;
  unsigned int upd_bf;
  unsigned int upd_2d_blocking;
  unsigned int upd_col_teams;
  unsigned int upd_row_teams;
  unsigned int ifm_subtasks;
  unsigned int ofm_subtasks;
  float alpha;
  float beta;
  cublasHandle_t cuda_handle;
} my_fc_bwd_config;

my_fc_fwd_config setup_my_fc_fwd(unsigned int N, unsigned int C, unsigned int K, unsigned int bn,
                                 unsigned int bc, unsigned int bk, my_eltwise_fuse fuse_type) {
  my_fc_fwd_config res;

  if ( fuse_type != MY_ELTWISE_FUSE_NONE ) {
    printf("this version of the code doesn't support any fusion!\n");
    exit(-1);
  }

  /* setting up some handle values */
  res.N = N;
  res.C = C;
  res.K = K;
  res.bn = bn;
  res.bc = bc;
  res.bk = bk;
  res.fuse_type = fuse_type;
  res.fwd_bf = 0;
  res.fwd_2d_blocking = 0;
  res.fwd_col_teams = 0;
  res.fwd_row_teams = 0;
  res.alpha = 1.0f;
  res.beta = 0.0f;
  cublasCreate( &(res.cuda_handle) );
  cublasSetMathMode( res.cuda_handle, CUBLAS_DEFAULT_MATH );

  return res;
}

my_fc_bwd_config setup_my_fc_bwd(unsigned int N, unsigned int C, unsigned int K, unsigned int bn,
                                 unsigned int bc, unsigned int bk, my_eltwise_fuse fuse_type) {
  my_fc_bwd_config res;

  if ( fuse_type != MY_ELTWISE_FUSE_NONE ) {
    printf("this version of the code doesn't support any fusion!\n");
    exit(-1);
  }

  /* setting up some handle values */
  res.N = N;
  res.C = C;
  res.K = K;
  res.bn = bn;
  res.bc = bc;
  res.bk = bk;
  res.fuse_type = fuse_type;

  /* setup parallelization strategy */
  res.bwd_bf = 0;
  res.bwd_2d_blocking = 0;
  res.bwd_col_teams = 0;
  res.bwd_row_teams = 0;
  res.upd_bf = 0;
  res.upd_2d_blocking = 0;
  res.upd_col_teams = 0;
  res.upd_row_teams = 0;
  res.ifm_subtasks = 0;
  res.ofm_subtasks = 0;
  res.alpha = 1.0f;
  res.beta = 0.0f;
  cublasCreate( &(res.cuda_handle) );
  cublasSetMathMode( res.cuda_handle, CUBLAS_DEFAULT_MATH );

  return res;
}

my_opt_config setup_my_opt(unsigned int C, unsigned int K, unsigned int bc, unsigned int bk,
                           float lr) {
  my_opt_config res;

  /* setting up some handle values */
  res.C = C;
  res.K = K;
  res.bc = bc;
  res.bk = bk;
  res.lr = lr;

  return res;
}

void my_fc_fwd_exec( my_fc_fwd_config cfg, const float* wt_ptr, const float* in_act_ptr, float* out_act_ptr,
                     const float* bias_ptr ) {
  cublasSgemm(cfg.cuda_handle, CUBLAS_OP_N, CUBLAS_OP_N, cfg.K, cfg.N, cfg.C,
              &(cfg.alpha), wt_ptr, cfg.K, in_act_ptr, cfg.C, &(cfg.beta), out_act_ptr, cfg.K);
}

void my_fc_bwd_exec( my_fc_bwd_config cfg, const float* wt_ptr, float* din_act_ptr,
                     float* dout_act_ptr, float* dwt_ptr, const float* in_act_ptr,
                     float* dbias_ptr,  my_pass pass ) {
  if( (pass & MY_PASS_BWD_D) == MY_PASS_BWD_D ) {
    cublasSgemm(cfg.cuda_handle, CUBLAS_OP_T, CUBLAS_OP_N, cfg.C, cfg.N, cfg.K,
                &(cfg.alpha), wt_ptr, cfg.K, dout_act_ptr, cfg.K, &(cfg.beta), din_act_ptr, cfg.C);
  }
  if( (pass & MY_PASS_BWD_W) == MY_PASS_BWD_W ) {
    cublasSgemm(cfg.cuda_handle, CUBLAS_OP_N, CUBLAS_OP_T, cfg.K, cfg.C, cfg.N,
                &(cfg.alpha), dout_act_ptr, cfg.K, in_act_ptr, cfg.C, &(cfg.beta), dwt_ptr, cfg.K);
  }
}

void my_opt_exec( my_opt_config cfg, float* wt_ptr, const float* delwt_ptr ) {
}

int main(int argc, char* argv[])
{
  float **act_host, **fil_host, **delact_host, **delfil_host;
  float **bias_host, **delbias_host;
  float **act_device, **fil_device, **delact_device, **delfil_device;
  float **bias_device, **delbias_device;
  my_eltwise_fuse my_fuse;
  my_fc_fwd_config* my_fc_fwd;
  my_fc_bwd_config* my_fc_bwd;
  my_opt_config* my_opt;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int MB = 256;          /* mini-batch size, "N" */
  int fuse_type = 0;      /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  int bn = 1;
  int bk = 1;
  int bc = 1;
  int *C;               /* number of input feature maps, "C" */
  int num_layers = 0;

  double l_total = 0.0;
  double gflop = 0.0;
  int i, j;
  double fil_size = 0.0;
  double act_size = 0.0;
  float lr = 0.2f;
  struct timeval l_start, l_end;

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters MB fuse_type type bn bk bc C1 C2 ... CN\n", argv[0]);
    return 0;
  }
  srand48(1);

  /* reading new values from cli */
  i = 1;
  num_layers = argc - 9;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) MB         = atoi(argv[i++]);
  if (argc > i) fuse_type  = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);
  if (argc > i) bn         = atoi(argv[i++]);
  if (argc > i) bk         = atoi(argv[i++]);
  if (argc > i) bc         = atoi(argv[i++]);
  /* allocate the number of channles buffer */
  if ( num_layers < 1 ) {
    printf("Usage: %s iters MB fuse_type type bn bk bc C1 C2 ... CN\n", argv[0]);
    return 0;
  }
  C = (int*)malloc((num_layers+2)*sizeof(int));
  for (j = 0 ; i < argc; ++i, ++j ) {
    C[j] = atoi(argv[i]);
  }
  /* handle softmax config */
  C[num_layers+1] = C[num_layers];

  if (type != 'A' && type != 'F' && type != 'B') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only)\n");
    return -1;
  }
  if ( fuse_type != 0 ) {
    printf("fuse type needs to be 0 (None)\n");
    return -1;
  }

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: N:%d\n", MB);
  printf("PARAMS: Layers: %d\n", num_layers);
  printf("PARAMS: ITERS:%d\n", iters);
  for (i = 0; i < num_layers; ++i ) {
    if (i == 0) {
      act_size += (double)(MB*C[i]*sizeof(float))/(1024.0*1024.0);
      printf("SIZE Activations  %i (%dx%d): %10.2f MiB\n", i, MB, C[i], (double)(MB*C[i]*sizeof(float))/(1024.0*1024.0) );
    }
    act_size += (double)(MB*C[i+1]*sizeof(float))/(1024.0*1024.0);
    fil_size += (double)(C[i]*C[i+1]*sizeof(float))/(1024.0*1024.0);
    printf("SIZE Filter       %i (%dx%d): %10.2f MiB\n", i, C[i], C[i+1], (double)(C[i]*C[i+1]*sizeof(float))/(1024.0*1024.0) );
    printf("SIZE Activations  %i (%dx%d): %10.2f MiB\n", i+1, MB, C[i+1], (double)(MB*C[i+1]*sizeof(float))/(1024.0*1024.0) );
  }
  act_size += (double)(MB*C[num_layers+1]*sizeof(float))/(1024.0*1024.0);
  printf("SIZE Activations softmax (%dx%d): %10.2f MiB\n", MB, C[num_layers+1], (double)(MB*C[num_layers+1]*sizeof(float))/(1024.0*1024.0) );
  printf("\nTOTAL SIZE Activations:    %10.2f MiB\n", act_size );
  printf("TOTAL SIZE Filter:         %10.2f MiB\n", fil_size );
  printf("TOTAL SIZE delActivations: %10.2f MiB\n", act_size );
  printf("TOTAL SIZE delFilter:      %10.2f MiB\n", fil_size );
  printf("TOTAL SIZE MLP:            %10.2f MiB\n", (2.0*fil_size) + (2.0*act_size) );

  cudaSetDevice(0);

  /* allocate data */
  /* +2 because of the softwax layer */
  act_host      = (float**)malloc( (num_layers+2)*sizeof(float*) );
  delact_host   = (float**)malloc( (num_layers+1)*sizeof(float*) );
  act_device    = (float**)malloc( (num_layers+2)*sizeof(float*) );
  delact_device = (float**)malloc( (num_layers+1)*sizeof(float*) );
  for ( i = 0 ; i < num_layers+2; ++i ) {
    act_host[i] = (float*)malloc( MB*C[i]*sizeof(float));
    cudaMalloc((void**)&(act_device[i]), MB*C[i]*sizeof(float));
    /* softmax has no incoming gradients */
    if ( i < num_layers+1 ) {
      delact_host[i] = (float*)malloc( MB*C[i]*sizeof(float) );
      cudaMalloc((void**)&(delact_device[i]), MB*C[i]*sizeof(float));
    }
  }
  fil_host      = (float**)malloc( num_layers*sizeof(float*) );
  delfil_host   = (float**)malloc( num_layers*sizeof(float*) );
  fil_device    = (float**)malloc( num_layers*sizeof(float*) );
  delfil_device = (float**)malloc( num_layers*sizeof(float*) );
  for ( i = 0 ; i < num_layers; ++i ) {
    fil_host[i]     = (float*)malloc( C[i]*C[i+1]*sizeof(float));
    delfil_host[i]  = (float*)malloc( C[i]*C[i+1]*sizeof(float));
    cudaMalloc((void**)&(fil_device[i]), C[i]*C[i+1]*sizeof(float));
    cudaMalloc((void**)&(delfil_device[i]), C[i]*C[i+1]*sizeof(float));
  }
  bias_host      = (float**)malloc( num_layers*sizeof(float*) );
  delbias_host   = (float**)malloc( num_layers*sizeof(float*) );
  bias_device    = (float**)malloc( num_layers*sizeof(float*) );
  delbias_device = (float**)malloc( num_layers*sizeof(float*) );
  for ( i = 0 ; i < num_layers; ++i ) {
    bias_host[i]               = (float*)malloc( C[i+1]*sizeof(float) );
    delbias_host[i]            = (float*)malloc( C[i+1]*sizeof(float) );
    cudaMalloc((void**)&(bias_device[i]), C[i+1]*sizeof(float));
    cudaMalloc((void**)&(delbias_device[i]), C[i+1]*sizeof(float));
  }

  /* init data */
  for ( i = 0 ; i < num_layers+2; ++i ) {
    my_init_buf( act_host[i], MB*C[i], 0, 0 );
  }
  for ( i = 0 ; i < num_layers+1; ++i ) {
    my_init_buf( delact_host[i], MB*C[i], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf( fil_host[i], C[i]*C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf( delfil_host[i], C[i]*C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf( bias_host[i], C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf( delbias_host[i], C[i+1], 0, 0 );
  }
  /* copying data to device */
  for ( i = 0 ; i < num_layers+2; ++i ) {
    cudaMemcpy(act_device[i], act_host[i], sizeof(float)*MB*C[i], cudaMemcpyHostToDevice);
  }
  for ( i = 0 ; i < num_layers+1; ++i ) {
    cudaMemcpy(delact_device[i], delact_host[i], sizeof(float)*MB*C[i], cudaMemcpyHostToDevice);
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    cudaMemcpy(fil_device[i], fil_host[i], sizeof(float)*C[i]*C[i+1], cudaMemcpyHostToDevice);
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    cudaMemcpy(delfil_device[i], delfil_host[i], sizeof(float)*C[i]*C[i+1], cudaMemcpyHostToDevice);
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    cudaMemcpy(bias_device[i], bias_host[i], sizeof(float)*C[i+1], cudaMemcpyHostToDevice);
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    cudaMemcpy(delbias_device[i], delbias_host[i], sizeof(float)*C[i+1], cudaMemcpyHostToDevice);
  }

  printf("\n");
  printf("##########################################\n");
  printf("#      Setting Up  (custom-Storage)      #\n");
  printf("##########################################\n");

  if ( fuse_type == 0 ) {
    my_fuse = MY_ELTWISE_FUSE_NONE;
#if 0
  } else if ( fuse_type == 1 ) {
    my_fuse = MY_ELTWISE_FUSE_BIAS;
  } else if ( fuse_type == 2 ) {
    my_fuse = MY_ELTWISE_FUSE_RELU;
  } else if ( fuse_type == 4 ) {
    my_fuse = MY_ELTWISE_FUSE_BIAS_RELU;
#endif
  } else {
    my_fuse = MY_ELTWISE_FUSE_NONE;
  }

  /* allocating handles */
  my_fc_fwd = (my_fc_fwd_config*) malloc( num_layers*sizeof(my_fc_fwd_config) );
  my_fc_bwd = (my_fc_bwd_config*) malloc( num_layers*sizeof(my_fc_bwd_config) );
  my_opt    = (my_opt_config*)    malloc( num_layers*sizeof(my_opt_config)    );

  /* setting up handles + scratch */
  for ( i = 0; i < num_layers; ++i ) {
    my_fc_fwd[i] = setup_my_fc_fwd(MB, C[i], C[i+1], (MB % bn == 0) ? bn : MB,
                                             (C[i  ] % bc == 0) ? bc : C[i  ],
                                             (C[i+1] % bk == 0) ? bk : C[i+1],
                                             my_fuse);

    my_fc_bwd[i] = setup_my_fc_bwd(MB, C[i], C[i+1], (MB % bn == 0) ? bn : MB,
                                             (C[i  ] % bc == 0) ? bc : C[i  ],
                                             (C[i+1] % bk == 0) ? bk : C[i+1],
                                             my_fuse);

    my_opt[i] = setup_my_opt( C[i], C[i+1], (C[i  ] % bc == 0) ? bc : C[i  ],
                                            (C[i+1] % bk == 0) ? bk : C[i+1],
                                            lr );

  }

  if ( type == 'F') {
    printf("##########################################\n");
    printf("#   Performance - FWD (custom-Storage)   #\n");
    printf("##########################################\n");
    gettimeofday(&l_start, NULL);
    for (j = 0; j < iters; ++j) {
      for ( i = 0; i < num_layers; ++i) {
        my_fc_fwd_exec( my_fc_fwd[i], fil_device[i], act_device[i], act_device[i+1],
                        bias_device[i] );
      }
    }
    cudaDeviceSynchronize();
    gettimeofday(&l_end, NULL);
    l_total = sec(l_start, l_end);

    gflop = 0.0;
    for ( i = 0; i < num_layers; ++i) {
      gflop += (2.0*(double)MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);
    printf("PERFDUMP,FP,%i,", MB );
    for ( i = 0; i < num_layers; ++i ) {
      printf("%i,", C[i] );
    }
    printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
  }

  if (type == 'B') {
    printf("##########################################\n");
    printf("#   Performance - BWD (custom-Storage)   #\n");
    printf("##########################################\n");
    gettimeofday(&l_start, NULL);
    for (j = 0; j < iters; ++j) {
      for ( i = num_layers-1; i > 0; --i) {
        my_fc_bwd_exec( my_fc_bwd[i], fil_device[i], delact_device[i], delact_device[i+1], delfil_device[i],
                        act_device[i], delbias_device[i], MY_PASS_BWD );
        my_opt_exec( my_opt[i], fil_device[i], delfil_device[i] );
      }
      my_fc_bwd_exec( my_fc_bwd[0], fil_device[0], delact_device[0], delact_device[0+1], delfil_device[0],
                      act_device[0], delbias_device[0], MY_PASS_BWD_W );
      my_opt_exec( my_opt[0], fil_device[0], delfil_device[0] );
    }
    cudaDeviceSynchronize();
    gettimeofday(&l_end, NULL);
    l_total = sec(l_start, l_end);

    gflop = 0.0;
    for ( i = num_layers-1; i > 0; --i) {
      gflop += (4.0*(double)MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    gflop += (2.0*(double)MB*(double)C[0]*(double)C[1]*(double)iters) / (1000.0*1000.0*1000.0);
    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);
    printf("PERFDUMP,BP,%i,", MB );
    for ( i = 0; i < num_layers; ++i ) {
      printf("%i,", C[i] );
    }
    printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
  }

  if (type == 'A') {
    printf("##########################################\n");
    printf("# Performance - FWD-BWD (custom-Storage) #\n");
    printf("##########################################\n");
    gettimeofday(&l_start, NULL);
    for (j = 0; j < iters; ++j) {
      for ( i = 0; i < num_layers; ++i) {
        my_fc_fwd_exec( my_fc_fwd[i], fil_device[i], act_device[i], act_device[i+1],
                        bias_device[i] );
      }
      for ( i = num_layers-1; i > 0; --i) {
        my_fc_bwd_exec( my_fc_bwd[i], fil_device[i], delact_device[i], delact_device[i+1], delfil_device[i],
                        act_device[i], delbias_device[i], MY_PASS_BWD );
        my_opt_exec( my_opt[i], fil_device[i], delfil_device[i] );
      }
      my_fc_bwd_exec( my_fc_bwd[0], fil_device[0], delact_device[0], delact_device[0+1], delfil_device[0],
                      act_device[0], delbias_device[0], MY_PASS_BWD_W );
      my_opt_exec( my_opt[0], fil_device[0], delfil_device[0] );
    }
    cudaDeviceSynchronize();
    gettimeofday(&l_end, NULL);
    l_total = sec(l_start, l_end);

    gflop = 0.0;
    for ( i = num_layers-1; i > 0; --i) {
      gflop += (6.0*(double)MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    gflop += (4.0*(double)MB*(double)C[0]*(double)C[1]*(double)iters) / (1000.0*1000.0*1000.0);
    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);
    printf("PERFDUMP,BP,%i,", MB );
    for ( i = 0; i < num_layers; ++i ) {
      printf("%i,", C[i] );
    }
    printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
  }

  /* deallocate data */
  for ( i = 0; i < num_layers; ++i ) {
    if ( i == 0 ) {
      free(act_host[i]);
      free(delact_host[i]);
      cudaFree(act_device[i]);
      cudaFree(delact_device[i]);
    }
    free(act_host[i+1]);
    free(delact_host[i+1]);
    cudaFree(act_device[i+1]);
    cudaFree(delact_device[i+1]);

    free(fil_host[i]);
    free(delfil_host[i]);
    free(bias_host[i]);
    free(delbias_host[i]);
    cudaFree(fil_device[i]);
    cudaFree(delfil_device[i]);
    cudaFree(bias_device[i]);
    cudaFree(delbias_device[i]);
  }
  free(act_host[num_layers+1]);
  cudaFree(act_device[num_layers+1]);

  free( my_opt );
  free( my_fc_fwd );
  free( my_fc_bwd );

  free( act_host );
  free( delact_host );
  free( fil_host );
  free( delfil_host );
  free( bias_host );
  free( delbias_host );

  free( act_device );
  free( delact_device );
  free( fil_device );
  free( delfil_device );
  free( bias_device );
  free( delbias_device );

  free( C );

  /* some empty lines at the end */
  printf("\n\n\n");

  return 0;
}

