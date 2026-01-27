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
      export CXX=`which CC`
      export CC=`which cc`
      export FC=`which ftn`
   else
      module load amdclang
   fi
fi
#export CXX=amdclang++
#export CC=amdclang
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/optimization/Allocations/3_memorypool

./umpire_setup.sh
export UMPIRE_PATH=${PWD}/Umpire_install
make
./memorypool
make clean

rm -rf Umpire_source Umpire_install
