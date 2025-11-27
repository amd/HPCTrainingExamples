#!/bin/bash

module load gcc
if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

rm -rf Chapter13
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
cd Chapter13/Kokkos/ShallowWater

mkdir hip_build; cd hip_build
cmake -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc ..
make -j 8 ShallowWater

./ShallowWater

cd ../../../..
rm -rf Chapter13
