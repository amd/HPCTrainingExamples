#!/bin/bash

if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
XNACK_COUNT=`rocminfo | grep xnack | wc -l`
if [ ${XNACK_COUNT} -lt 1 ]; then
   echo "Skip"
else
   module load amdclang
   module load kokkos

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   pushd ${REPO_DIR}/ManagedMemory/Kokkos_Code

   # To run with managed memory
   export HSA_XNACK=1

   rm -rf build
   mkdir build && cd build
   CXX=hipcc cmake ..
   make
   ./kokkos_code

   cd ..
   rm -rf build
   popd

fi
