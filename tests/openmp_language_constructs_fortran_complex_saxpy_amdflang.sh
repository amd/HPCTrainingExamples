#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran
cd complex_saxpy
make
./complex_saxpy

make clean
