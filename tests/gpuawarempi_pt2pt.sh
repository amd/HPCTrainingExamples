#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
GPU_COUNT=`rocminfo | grep "Device Type:             GPU"  | wc -l`
if [ ${GPU_COUNT} -lt 2 ]; then
   echo "Skip"
else
   module load openmpi
   export OMPI_CXX=hipcc

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/MPI-examples
   mpicxx -o ./pt2pt ./pt2pt.cpp

   mpirun -n 2 ./pt2pt

   make clean
fi
