#!/bin/bash

module load rocm
module load amdflang-new

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/3_reduction/2_reduction_solution

make
./freduce
make clean
