#!/bin/bash

module load rocm

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/hip-stream

rm -rf build
mkdir build && cd build
cmake ..
make
./stream
cd ..
rm -rf build
