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

   PROB_NAME=programming_model_raja_code
   rm -rf ${PROB_NAME}
   mkdir ${PROB_NAME} && cd ${PROB_NAME}

   PWDir=`pwd`

   git clone --recursive https://github.com/LLNL/RAJA.git Raja_build
   cd Raja_build

   rm -rf build_hip
   mkdir build_hip && cd build_hip

   cmake -DCMAKE_INSTALL_PREFIX=${PWDir}/Raja_HIP \
         -DROCM_ROOT_DIR=/opt/rocm \
         -DHIP_ROOT_DIR=/opt/rocm \
         -DHIP_PATH=/opt/rocm/bin \
         -DENABLE_TESTS=Off \
         -DENABLE_EXAMPLES=Off \
         -DRAJA_ENABLE_EXERCISES=Off \
         -DENABLE_HIP=On \
         ..

   make -j 8
   make install

   cd ../..

   rm -rf Raja_build

   export Raja_DIR=${PWDir}/Raja_HIP

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/ManagedMemory/Raja_Code

   # To run with managed memory
   export HSA_XNACK=1

   rm -rf build
   mkdir build && cd build
   CXX=hipcc Raja_DIR=${PWDir}/Raja_HIP cmake ..
   make
   ./raja_code

   cd ..
   rm -rf build

   cd ${PWDir}
   rm -rf Raja_HIP

   cd ..
   rm -rf ${PROB_NAME}
fi
