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
      # For Cray system, set compiler variables
      export CXX=`which CC`
      export CC=`which cc`
      export FC=`which ftn`
   fi
   if [[ "`module list |& grep -w PrgEnv-cray |wc -l`" -lt 1 ]]; then
      # For AMD compilers, allow override
      if [[ "`module avail |& grep -w amdflang-new |wc -l`" -ge 1 ]]; then
         if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
            module unload rocm
            module switch amd amdflang-new
         else
            module load amdflang-new
         fi
      else
         module load amdclang
      fi
   fi

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/memory_pragmas

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
