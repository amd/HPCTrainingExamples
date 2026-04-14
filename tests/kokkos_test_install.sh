#!/bin/bash

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load amdclang
module load kokkos

GFX_MODEL=`rocminfo | grep gfx | sed -e 's/Name://' | head -1 |sed 's/ //g'`

CLONE_DIR=$(mktemp -d -p "$(pwd)" Chapter13_XXXXXX)
trap "rm -rf ${CLONE_DIR}" EXIT
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 ${CLONE_DIR}
pushd ${CLONE_DIR}/Kokkos/StreamTriad/Ver3
sed -i '/cmake_minimum_required/a\
cmake_policy(SET CMP0074 NEW)' CMakeLists.txt
sed -i '/project (StreamTriad)/a\
set(GPU_TARGETS "${GFX_MODEL}" CACHE STRING "GPU targets" FORCE)\
set(AMDGPU_TARGETS "${GFX_MODEL}" CACHE STRING "AMD GPU targets" FORCE)' CMakeLists.txt
sed -i -e 's/80000000/100000/' StreamTriad.cc

rm -rf build
mkdir build && cd build
cmake ..
make
HSA_XNACK=1 ./StreamTriad

popd
