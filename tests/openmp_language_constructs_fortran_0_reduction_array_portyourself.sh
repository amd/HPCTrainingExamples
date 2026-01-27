#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load amdflang-new >& /dev/null
if [ "$?" == "1" ]; then
   if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
      export CXX=${ROCM_PATH}/llvm/bin/amdclang++
      export CC=${ROCM_PATH}/llvm/bin/amdclang
      export FC=${ROCM_PATH}/llvm/bin/amdflang
   else
      module load amdclang
   fi
fi
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/9_reduction_array/0_reduction_scalar_portyourself
make
./reduction_array

make clean
