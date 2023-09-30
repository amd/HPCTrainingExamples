#!/bin/bash

module load gcc
module load rocm
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
cd Chapter13/Kokkos/ShallowWater

mkdir serial_build; cd serial_build
cmake ..
make -j 8 ShallowWater_par5

./ShallowWater_par5

cd ../../../..
rm -rf Chapter13
