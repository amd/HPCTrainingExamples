#!/bin/bash

export HSA_XNACK=1

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   module switch PrgEnv-cray PrgEnv-amd
   export CXX=${ROCM_PATH}/llvm/bin/amdclang++
else
   module load amdclang
fi

module load kokkos 

GFX_MODEL=`rocminfo | grep gfx | sed -e 's/Name://' | head -1 |sed 's/ //g'`

CLONE_DIR=$(mktemp -d -p /tmp Chapter13_XXXXXX)
trap "rm -rf ${CLONE_DIR}" EXIT
git clone --depth=1 --recurse-submodules --shallow-submodules https://github.com/EssentialsOfParallelComputing/Chapter13 ${CLONE_DIR}
pushd ${CLONE_DIR}/Kokkos/ShallowWater

mkdir hip_build
sed -i '/cmake_minimum_required/a\
cmake_policy(SET CMP0074 NEW)' CMakeLists.txt
sed -i "/project (ShallowWater)/a\\
set(GPU_TARGETS \"${GFX_MODEL}\" CACHE STRING \"GPU targets\" FORCE)\\
set(AMDGPU_TARGETS \"${GFX_MODEL}\" CACHE STRING \"AMD GPU targets\" FORCE)" CMakeLists.txt
sed -i 's/add_subdirectory(Kokkos)/find_package(Kokkos REQUIRED)/' CMakeLists.txt
cd hip_build
# Pin OpenMP to the host libomp.so. find_package(Kokkos) pulls in
# OpenMP::OpenMP_CXX (the OpenMP-enabled Kokkos backend); under the Cray CC
# wrapper + ROCm clang CMake's FindOpenMP mis-resolves it to the amdgcn device
# archive libompdevice.a, which ld.lld then rejects on the host link
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
make -j ShallowWater_par2

./ShallowWater_par2

popd
