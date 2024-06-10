#!/bin/bash

module load rocm

cd ${REPO_DIR}/HIP/hip-stream

mkdir build && cd build
cmake ..
make
./stream
cd ..
rm -rf build
