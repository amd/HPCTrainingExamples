#!/bin/bash

module load amdclang
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/4_reduction_scalar/1_reduction_scalar_solution
cd reduction_scalar
make
./reduction_scalar

make clean
