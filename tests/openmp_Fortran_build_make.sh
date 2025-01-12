#!/bin/bash

module load rocm
XNACK_COUNT=`rocminfo | grep xnack | wc -l`
if [ ${XNACK_COUNT} -lt 1 ]; then
   echo "Skip"
else
   export HSA_XNACK=1

   module load amdflang-new-beta-drop

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/BuildExamples
   make openmp_code
   ./openmp_code

   make clean
fi
