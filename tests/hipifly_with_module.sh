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
   module -t list 2>&1 | grep -q "^rocm"
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

module load hipifly

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT

cd ${BUILD_DIR}

cp -a ${REPO_DIR}/hipifly/ .
pushd hipifly/vector_add
rm src/hipifly.h

make DFLAGS="-DENABLE_HIP -I${HIPIFLY_PATH} -fPIE"
./vector_add
