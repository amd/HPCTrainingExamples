#!/bin/bash

module load gcc
module load rocm
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
cd Chapter13/Kokkos/ShallowWater
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

mkdir openmp_build; cd openmp_build
cmake -DKokkos_ENABLE_OPENMP=ON ..
make -j 8 ShallowWater

./ShallowWater
ls -l

cd ../../../..
#rm -rf Chapter13
