#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPStdPar/CXX/ShallowWater_Orig

if [[ "`module list |& grep PrgEnv-cray | wc -l`" -ge 1 ]]; then
   export CXX=`which CC`
fi

rm -rf build
mkdir build && cd build
cmake ..
make
./ShallowWater

cd ..
rm -rf build
