#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran
cd reduction_scalar
make
./reduction_scalar

make clean
