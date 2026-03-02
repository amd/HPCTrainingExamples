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

module load hipfort
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRC_DIR=${REPO_DIR}/HIPFort/hipgemm

BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cp ${SRC_DIR}/Makefile ${SRC_DIR}/*.f90 ${BUILD_DIR}/
cd ${BUILD_DIR}

make gemm_local
./gemm_local
