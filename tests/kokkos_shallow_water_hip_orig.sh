#!/bin/bash

module load gcc
module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

PROB_NAME=kokkos_shallow_water_hip_orig
rm -rf ${PROB_NAME}
mkdir ${PROB_NAME} && cd ${PROB_NAME}

git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
cd Chapter13/Kokkos/ShallowWater

mkdir hip_build; cd hip_build
cmake -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc ..
make -j 8 ShallowWater

./ShallowWater

cd ../../../../..
rm -rf ${PROB_NAME}
