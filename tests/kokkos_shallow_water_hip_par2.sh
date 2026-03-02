#!/bin/bash

export HSA_XNACK=1
module load gcc
module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

PROB_NAME=kokkos_shallow_water_hip_par2
rm -rf ${PROB_NAME}
mkdir ${PROB_NAME} && cd ${PROB_NAME}

git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
cd Chapter13/Kokkos/ShallowWater

mkdir hip_build; cd hip_build
cmake -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc ..
make -j 8 ShallowWater_par2

./ShallowWater_par2

cd ../../../../..
rm -rf ${PROB_NAME}
