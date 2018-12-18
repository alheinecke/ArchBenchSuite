#!/bin/bash

# 1st socket
OMP_NUM_THREADS=18 KMP_AFFINITY=proclist=[0-17],granularity=fine,explicit,norespect LD_PRELOAD=libhugetlbfs.so HUGETLB_MORECORE=yes ./sgemm.exe 9000 9000 9000 n n 10 10 &

# 2nd socket
OMP_NUM_THREADS=18 KMP_AFFINITY=proclist=[18-35],granularity=fine,explicit,norespect LD_PRELOAD=libhugetlbfs.so HUGETLB_MORECORE=yes ./sgemm.exe 9000 9000 9000 n n 10 10 &

