#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   if [[ "`module list |& grep PrgEnv-cray | wc -l`" -ge 1 ]]; then
      export CXX=`which CC`
      export CC=`which cc`
   elif [[ "`module list |& grep PrgEnv-amd | wc -l`" -ge 1 ]]; then
      export CXX=${ROCM_PATH}/llvm/bin/amdclang++
      export CC=${ROCM_PATH}/llvm/bin/amdclang
   fi
else
   module load amdclang
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/2_vecadd/2_vecadd_map

make
./vecadd
make clean
