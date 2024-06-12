#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPStdPar/CXX/ShallowWater_Orig

rm -rf build
mkdir build && cd build
cmake ..
make
./ShallowWater

cd ..
rm -rf build
