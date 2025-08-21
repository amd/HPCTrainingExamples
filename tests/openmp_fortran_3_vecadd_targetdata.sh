#!/bin/bash

module load rocm
module load amdflang-new

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/2_vecadd/3_vecadd_targetdata

make
./vecadd
make clean
