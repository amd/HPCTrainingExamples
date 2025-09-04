#!/bin/bash

module load gcc
module load rocm

rm -rf Chapter13
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
cd Chapter13/Kokkos/ShallowWater

mkdir serial_build; cd serial_build
cmake ..
make -j 8 ShallowWater

./ShallowWater

cd ../../../..
rm -rf Chapter13
