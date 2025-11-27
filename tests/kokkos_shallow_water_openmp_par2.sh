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
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_PROC_BIND=true

mkdir openmp_build; cd openmp_build
cmake -DKokkos_ENABLE_OPENMP=ON ..
make -j 8 ShallowWater_par2

./ShallowWater_par2
ls -l

cd ../../../..
rm -rf Chapter13
