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

module load kokkos

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
   # Pin OpenMP to the host libomp.so. find_package(Kokkos) pulls in
   # OpenMP::OpenMP_CXX (the OpenMP-enabled Kokkos backend); under the Cray CC
   # wrapper + ROCm clang CMake's FindOpenMP mis-resolves it to the amdgcn
   # device archive libompdevice.a, which ld.lld then rejects on the host link
   # ("incompatible with elf64-x86-64"). Feed host libomp.so to FindOpenMP.
   if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
     OMP_CXX="${CXX:-$(command -v CC)}"
   else
     OMP_CXX="${CXX:-$(command -v amdclang++ || command -v clang++)}"
   fi
   OMP_HOST_LIB="$(${OMP_CXX} -print-file-name=libomp.so 2>/dev/null)"
   OMP_HINTS=()
   if [ -n "${OMP_HOST_LIB}" ] && [ -f "${OMP_HOST_LIB}" ]; then
     OMP_HINTS=(
       -DOpenMP_CXX_FLAGS="-fopenmp=libomp"
       -DOpenMP_CXX_LIB_NAMES="omp"
       -DOpenMP_omp_LIBRARY="${OMP_HOST_LIB}"
     )
   fi
   cmake .. "${OMP_HINTS[@]}"
   make
   ./kokkos_code

fi
