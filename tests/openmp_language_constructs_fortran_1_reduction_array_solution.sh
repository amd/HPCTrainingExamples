#!/bin/bash

module load rocm
module load amdflang-new
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/9_reduction_array/1_reduction_array_solution
make
./reduction_array

make clean
