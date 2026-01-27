#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
XNACK_COUNT=`rocminfo | grep xnack | wc -l`
if [ ${XNACK_COUNT} -lt 1 ]; then
   echo "Skip"
else
   export HSA_XNACK=1
   if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
      export CXX=${ROCM_PATH}/llvm/bin/amdclang++
   else
      module load amdclang
   fi

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/HIP-OpenMP/CXX/daxpy
   make
   ./daxpy

   make clean
fi
