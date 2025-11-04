#!/bin/bash

module load rocm
XNACK_COUNT=`rocminfo | grep xnack | wc -l`
if [ ${XNACK_COUNT} -lt 1 ]; then
   echo "Skip"
else
   export HSA_XNACK=1

   module load amdflang-new
   if [ "$?" == "1" ]; then
      module load amdclang
   fi

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/BuildExamples
   mkdir build && cd build && cmake -DCMAKE_Fortran_COMPILER=`which amdflang` ..
   make openmp_code
   ./openmp_code

   make clean
   cd ..
   rm -rf build
fi
