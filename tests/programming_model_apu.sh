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
      export CXX=`which CC`
   else
      module load amdclang
   fi

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/ManagedMemory/APU_Code
   export HSA_XNACK=1
   make gpu_code
   ./gpu_code

   make clean
fi
