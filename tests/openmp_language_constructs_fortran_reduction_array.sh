#!/bin/bash

module load amdclang
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/4_reduction_array/1_reduction_array_solution
cd reduction_array
make
./reduction_array

make clean
