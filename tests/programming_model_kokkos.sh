#!/bin/bash

if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
   if [ -z "$HIPCC" ]; then
      export HIPCC=`which hipcc`
   fi
else
   module list 2>&1 | grep -q -w "rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
fi

XNACK_COUNT=`rocminfo | grep xnack | wc -l`
if [ ${XNACK_COUNT} -lt 1 ]; then
   echo "Skip"
else

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   SRC_DIR=${REPO_DIR}/ManagedMemory/Kokkos_Code

   BUILD_DIR=$(mktemp -d)
   trap "rm -rf ${BUILD_DIR}" EXIT
   cp ${SRC_DIR}/* ${BUILD_DIR}/

   # To run with managed memory
   export HSA_XNACK=1

   cd ${BUILD_DIR}
   mkdir build && cd build
   CXX=hipcc cmake ..
   make
   ./kokkos_code

fi
