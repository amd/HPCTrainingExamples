#!/bin/bash

module load gcc
if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

rm -rf Chapter13
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
pushd Chapter13/Kokkos/ShallowWater

mkdir serial_build; cd serial_build
cmake ..
make -j 8 ShallowWater_par3

./ShallowWater_par3
cd ..

popd
rm -rf Chapter13
