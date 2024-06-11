#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C
cd reduction_scalar
make
./reduction_scalar

make clean
