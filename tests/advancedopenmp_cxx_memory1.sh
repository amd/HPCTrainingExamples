#!/bin/bash

if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
GFX_MODEL=`rocminfo | grep gfx | sed -e 's/Name://' | head -1 |sed 's/ //g'`
if [ "${GFX_MODEL}" = "gfx1030" ] ; then
   echo "Skip"
else
   module load amdclang

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/memory_pragmas

   export LIBOMPTARGET_INFO=-1
   export LIBOMPTARGET_INFO_SUPPORT=0
   export SLURM_BATCH_WAIT=0
   export OMP_TARGET_OFFLOAD=MANDATORY

   rm -rf build
   mkdir build && cd build
   cmake ..
   make mem1
   ./mem1

   cd ..
   rm -rf build
fi
