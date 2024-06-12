#!/bin/bash
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/memory_pragmas

export LIBOMPTARGET_INFO=-1
export OMP_TARGET_OFFLOAD=MANDATORY

rm -rf build
mkdir build && cd build
cmake ..
make mem1
./mem1

cd ..
rm -rf build
