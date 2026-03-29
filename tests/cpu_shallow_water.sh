#!/bin/bash

module load gcc
module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi


CLONE_DIR=$(mktemp -d -p "$(pwd)" Chapter13_XXXXXX)
trap "rm -rf ${CLONE_DIR}" EXIT
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 ${CLONE_DIR}
pushd ${CLONE_DIR}/Kokkos/ShallowWater

sed -i '/cmake_minimum_required/a\
cmake_policy(SET CMP0074 NEW)' CMakeLists.txt

mkdir cpu_build; cd cpu_build
cmake .. 
make -j 8 ShallowWater

./ShallowWater

popd
