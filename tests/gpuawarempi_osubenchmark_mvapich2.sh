#!/bin/bash

module purge

module load rocm mvapich2/2.3.7

export HIP_VISIBLE_DEVICES=0,1

export MV2_USE_ROCM=1

# NOTE: since mvapich2 needs a hostfile to run on multiple nodes
# this benchmark is run only on the local host unlike the openmpi
# analog (i.e. gpuawarempi_osubenchmark_openmpi.sh) where we specify
# -N 2 in the run command

mpiexec -np 2 $MV2_PATH/libexec/osu-microbenchmarks/mpi/collective/osu_allreduce -d rocm

