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

   if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
      export CXX=${ROCM_PATH}/llvm/bin/amdclang++
      export CC=${ROCM_PATH}/llvm/bin/amdclang
      export FC=${ROCM_PATH}/llvm/bin/amdflang
   else
      module load amdclang
   fi

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/atomics_openmp

   make arraysum10
   export HSA_XNACK=1
   ./arraysum10

   make clean
fi

