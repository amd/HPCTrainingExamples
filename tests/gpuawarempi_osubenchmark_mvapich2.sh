#!/bin/bash



module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   export MPIRUN=srun
else
   export MPIRUN=mpiexec
   module load mvapich2/2.3.7
fi

export HIP_VISIBLE_DEVICES=0,1

export MV2_USE_ROCM=1

# NOTE: since mvapich2 needs a hostfile to run on multiple nodes
# this benchmark is run only on the local host unlike the openmpi
# analog (i.e. gpuawarempi_osubenchmark_openmpi.sh) where we specify
# -N 2 in the run command

${MPIRUN} -np 2 $MV2_PATH/libexec/osu-microbenchmarks/mpi/collective/osu_allreduce -d rocm

