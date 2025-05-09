#!/bin/bash

module load amdflang-new

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/3_reduction/0_reduction_portyourself

make
./freduce
make clean
