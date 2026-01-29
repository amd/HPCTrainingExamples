#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
XNACK_COUNT=`rocminfo | grep xnack | wc -l`
if [ ${XNACK_COUNT} -lt 1 ]; then
   echo "Skip"
else
   export HSA_XNACK=1

   if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
      # For Cray system, set compiler variables
      export CXX=`which CC`
      export CC=`which cc`
      export FC=`which ftn`
   fi
   if [[ "`module list |grep -w PrgEnv-cray |wc -l`" -lt 1 ]]; then
      # For AMD compilers, allow override
      if [[ "`module avail |grep -w amdflang-new |wc -l`" -ge 1 ]]; then
         if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
            module switch amd amdflang-new
         else
            module load amdflang-new
         fi
      else
         module load amdclang
      fi
   fi
      

   if [[ "`printenv |grep -w CRAY |wc -l`" -lt 1 ]]; then
      # Not on a Cray system
      module load amdflang-new >& /dev/null
      if [ "$?" == "1" ]; then
         module load amdclang
      fi
   else
   fi

      if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/BuildExamples
   mkdir build && cd build && cmake -DCMAKE_Fortran_COMPILER=`which amdflang` ..
   make openmp_code
   ./openmp_code

   make clean
   cd ..
   rm -rf build
fi
