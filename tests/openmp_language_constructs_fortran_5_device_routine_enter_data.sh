#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

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

export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/6_device_routines/device_routine_with_interface/5_device_routine_enter_data
make
./device_routine

make clean
