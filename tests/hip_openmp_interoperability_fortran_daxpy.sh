#!/bin/bash

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
   export HIP_PLATFORM=amd
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

XNACK_COUNT=`rocminfo | grep xnack | wc -l`
if [ ${XNACK_COUNT} -lt 1 ]; then
   echo "Skip"
else
   export HSA_XNACK=1

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/HIP-OpenMP/F/daxpy

   SRC_DIR=$(pwd)
   BUILD_DIR=$(mktemp -d)
   trap "rm -rf ${BUILD_DIR}" EXIT
   cp * ${BUILD_DIR}

   cd ${BUILD_DIR}

   # ROCm 7.2.4's OpenMP-offload link goes through clang-linker-wrapper, which
   # (unlike 7.2.3 and plain flang/clang++) does not honor the -L${ROCM_PATH}/lib
   # passed by the Cray ftn/CC wrapper. Put the ROCm lib dir on LIBRARY_PATH so
   # the offload linker can resolve the host library libamdhip64.
   export LIBRARY_PATH=${ROCM_PATH}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}

   make
   ./daxpy
fi
