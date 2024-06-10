#!/bin/bash

module load amdclang

cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/vecadd

make
./vecadd
make clean
