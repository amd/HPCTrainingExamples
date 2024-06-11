#!/bin/bash

module load rocm

cd ${REPO_DIR}/HIP/vectorAdd

mkdir build && cd build
cmake ..
make
./vectoradd
cd ..
rm -rf build
