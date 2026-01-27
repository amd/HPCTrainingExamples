#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
GFX_MODEL=`rocminfo | grep gfx | sed -e 's/Name://' | head -1 |sed 's/ //g'`
if [ "${GFX_MODEL}" = "gfx1030" ] ; then
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
   cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/memory_pragmas

   export LIBOMPTARGET_INFO=-1
   export LIBOMPTARGET_INFO_SUPPORT=0
   export SLURM_BATCH_WAIT=0
   export OMP_TARGET_OFFLOAD=MANDATORY

   rm -rf build
   mkdir build && cd build
   cmake ..
   make mem2
   ./mem2

   cd ..
   rm -rf build
fi
