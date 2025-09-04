#!/bin/bash

export HSA_XNACK=1
module load gcc
module load rocm

rm -rf Chapter13
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
cd Chapter13/Kokkos/ShallowWater

mkdir hip_build; cd hip_build
cmake -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc ..
make -j 8 ShallowWater_par2

./ShallowWater_par2

cd ..
rm -rf hip_build

cd ../../..
rm -rf Chapter13
