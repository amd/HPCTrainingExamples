#!/bin/bash

module load amdflang-new

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/1_saxpy/1_saxpy_omptarget

make
./saxpy
make clean
