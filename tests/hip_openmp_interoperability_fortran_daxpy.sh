#!/bin/bash

module load rocm
XNACK_COUNT=`rocminfo | grep xnack | wc -l`
if [ ${XNACK_COUNT} -lt 1 ]; then
   echo "Skip"
else
   export HSA_XNACK=1
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/HIP-OpenMP/F/daxpy
   make
   ./daxpy

   make clean
fi
