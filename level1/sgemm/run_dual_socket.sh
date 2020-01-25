#!/bin/bash

#OMP_NUM_THREADS=28 KMP_AFFINITY=proclist=[0-27],granularity=fine,explicit,norespect LD_PRELOAD=libhugetlbfs.so HUGETLB_MORECORE=yes ./sgemm.exe 9000 9000 9000 n n 10 1000 &
#OMP_NUM_THREADS=28 KMP_AFFINITY=proclist=[28-55],granularity=fine,explicit,norespect LD_PRELOAD=libhugetlbfs.so HUGETLB_MORECORE=yes ./sgemm.exe 9000 9000 9000 n n 10 1000 &

OMP_NUM_THREADS=28 KMP_AFFINITY=proclist=[0-27],granularity=fine,explicit,norespect ./sgemm.exe 9000 9000 9000 n n 10 1000 &
OMP_NUM_THREADS=28 KMP_AFFINITY=proclist=[28-55],granularity=fine,explicit,norespect ./sgemm.exe 9000 9000 9000 n n 10 1000 &

