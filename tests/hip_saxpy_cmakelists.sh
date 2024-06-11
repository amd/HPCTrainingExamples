#!/bin/bash

module load rocm

cd ${REPO_DIR}/HIP/saxpy

mkdir build && cd build
cmake ..
make
./saxpy
cd ..
rm -rf build
