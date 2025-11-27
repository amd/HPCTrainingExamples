#!/bin/bash

if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load amdclang
module load kokkos

rm -rf Chapter13
git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
pushd Chapter13/Kokkos/StreamTriad/Ver1
sed -i -e 's/80000000/100000/' StreamTriad.cc

rm -rf build
mkdir build && cd build
CXX=hipcc cmake ..
make
./StreamTriad

popd
rm -rf Chapter13
