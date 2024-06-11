#!/bin/bash

module load rocm

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/saxpy

rm -rf build
mkdir build && cd build
cmake ..
make
./saxpy
cd ..
rm -rf build
