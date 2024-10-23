#!/bin/bash

module load clang/15

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/2_reduction/1_reduction_solution

make
./freduce
make clean
