#!/bin/bash

module load gcc
module load rocm

rm -rf Chapter13
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
cd Chapter13/Kokkos/ShallowWater
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_PROC_BIND=true

mkdir openmp_build; cd openmp_build
cmake -DKokkos_ENABLE_OPENMP=ON ..
make -j 8 ShallowWater_par4

./ShallowWater_par4
ls -l

cd ../../../..
rm -rf Chapter13
