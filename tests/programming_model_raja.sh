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

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   PWDir=`pwd`

   SRC_DIR=$(pwd)
   BUILD_DIR=$(mktemp -d)
   trap "rm -rf ${BUILD_DIR}" EXIT

   cd ${BUILD_DIR}

   PROB_NAME=programming_model_raja_code
   mkdir ${PROB_NAME} && cd ${PROB_NAME}

   git clone --recursive --depth 1 --shallow-submodules https://github.com/LLNL/RAJA.git Raja_build
   cd Raja_build

   rm -rf build_hip
   mkdir build_hip && cd build_hip

   export Raja_DIR=${BUILD_DIR}/Raja_HIP

   cmake -DCMAKE_INSTALL_PREFIX=${Raja_DIR}/Raja_HIP \
         -DROCM_ROOT_DIR=${ROCM_PATH} \
         -DHIP_ROOT_DIR=${ROCM_PATH} \
         -DHIP_PATH=${ROCM_PATH}/bin \
         -DENABLE_TESTS=Off \
         -DENABLE_EXAMPLES=Off \
         -DRAJA_ENABLE_EXERCISES=Off \
         -DENABLE_HIP=On \
         ..

   make -j 8
   make install

   cd ../..

   rm -rf Raja_build || true

   # To run with managed memory
   export HSA_XNACK=1

   cd ${BUILD_DIR}
   rm -rf raja_example_build || true
   mkdir -p raja_example_build && cd raja_example_build
   CXX=hipcc Raja_DIR=${Raja_DIR}/Raja_HIP cmake ${REPO_DIR}/ManagedMemory/Raja_Code
   make
   ./raja_code
fi
