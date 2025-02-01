#!/bin/bash

module load amdclang
module load rocm
module load kokkos

git clone --recursive https://github.com/EssentialsOfParallelComputing/Chapter13 Chapter13
pushd Chapter13/Kokkos/StreamTriad/Ver4
sed -i -e 's/80000000/100000/' StreamTriad.cc

rm -rf build
mkdir build && cd build
CXX=hipcc cmake ..
make
./StreamTriad

popd
rm -rf Chapter13
