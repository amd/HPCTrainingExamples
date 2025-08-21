#!/bin/bash

module load rocm
module load amdflang-new
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/3_reduction/2_reduction_solution_usm

make
./freduce
make clean
