#!/bin/bash

module load rocm
module load amdflang-new

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/1_saxpy/0_saxpy_portyourself

make
./saxpy
make clean
