#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C
cd reduction_array
make
./reduction_array

make clean
