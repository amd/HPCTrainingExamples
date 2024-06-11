#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran
cd reduction_array
make
./reduction_array

make clean
