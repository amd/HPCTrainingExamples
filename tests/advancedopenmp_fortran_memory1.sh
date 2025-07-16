#!/bin/bash

module load rocm
GFX_MODEL=`rocminfo | grep gfx | sed -e 's/Name://' | head -1 |sed 's/ //g'`
if [ "${GFX_MODEL}" = "gfx1030" ] ; then
   echo "Skip"
else
   module load amdflang-new

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/memory_pragmas

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
