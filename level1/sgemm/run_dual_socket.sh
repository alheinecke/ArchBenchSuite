#!/bin/bash

# 1st socket
OMP_NUM_THREADS=28 KMP_AFFINITY=proclist=[0-27],granularity=fine,explicit srun ./sgemm.exe 9000 9000 9000 n n 10 10 &

# 2nd socket
OMP_NUM_THREADS=28 KMP_AFFINITY=proclist=[28-55],granularity=fine,explicit srun ./sgemm.exe 9000 9000 9000 n n 10 10 &

