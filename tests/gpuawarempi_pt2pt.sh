#!/bin/bash

module load rocm
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
