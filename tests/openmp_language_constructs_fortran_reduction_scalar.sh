#!/bin/bash

module load amdflang-new
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/4_reduction_scalars/1_reduction_scalar_solution
make
./reduction_scalar

make clean
