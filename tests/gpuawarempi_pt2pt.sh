#!/bin/bash

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
fi

GPU_COUNT=`rocminfo | grep "Device Type:             GPU"  | wc -l`
if [ ${GPU_COUNT} -lt 2 ]; then
   echo "Skip"
else
   if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
      MPIRUN=srun
      MPICXX=CC
   else
      module load openmpi
      MPIRUN=mpirun
      export OMPI_CXX=hipcc
      MPICXX=mpicxx
   fi

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/MPI-examples
   ${MPICXX} -o ./pt2pt ./pt2pt.cpp -I${ROCM_PATH}/include

   ${MPIRUN} -n 2 ./pt2pt

   rm -f pt2pt
fi
