#!/bin/bash
module load amdclang

cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/memory_pragmas

export LIBOMPTARGET_INFO=-1
export OMP_TARGET_OFFLOAD=MANDATORY

mkdir build && cd build
cmake ..
make mem2
./mem2

cd ..
rm -rf build
