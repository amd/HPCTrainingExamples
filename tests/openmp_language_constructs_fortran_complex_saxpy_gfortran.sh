#!/bin/bash

module load amd-gcc
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran
cd complex_saxpy
make
./complex_saxpy

make clean
