#!/bin/bash

module load rocm
module load amdflang-new

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/1_saxpy/6_saxpy_targetdata

make
./saxpy
make clean
