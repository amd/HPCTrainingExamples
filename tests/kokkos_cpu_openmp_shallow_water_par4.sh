#!/bin/bash

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
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

sed -i 's/Kokkos::View<double \*\*>/Kokkos::View<double **, Kokkos::OpenMP>/g' ShallowWater_par4.cc
sed -i 's/Kokkos::MDRangePolicy<Kokkos::Rank<2>>/Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>/g' ShallowWater_par4.cc
sed -i 's/Kokkos::RangePolicy<>/Kokkos::RangePolicy<Kokkos::OpenMP>/g' ShallowWater_par4.cc
sed -i 's/KOKKOS_LAMBDA/[=]/g' ShallowWater_par4.cc

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

sed -i '/cmake_minimum_required/a\
cmake_policy(SET CMP0074 NEW)' CMakeLists.txt
sed -i "/project (ShallowWater)/a\\
set(GPU_TARGETS \"${GFX_MODEL}\" CACHE STRING \"GPU targets\" FORCE)\\
set(AMDGPU_TARGETS \"${GFX_MODEL}\" CACHE STRING \"AMD GPU targets\" FORCE)" CMakeLists.txt
sed -i 's/add_subdirectory(Kokkos)/find_package(Kokkos REQUIRED)/' CMakeLists.txt

mkdir openmp_build && cd openmp_build
cmake ..
make -j ShallowWater_par4

./ShallowWater_par4

popd
