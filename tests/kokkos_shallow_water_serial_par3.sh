#!/bin/bash

module load gcc
module load rocm
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
pushd Chapter13/Kokkos/ShallowWater

mkdir serial_build; cd serial_build
cmake ..
make -j 8 ShallowWater_par3

./ShallowWater_par3
cd ..

popd
rm -rf Chapter13
