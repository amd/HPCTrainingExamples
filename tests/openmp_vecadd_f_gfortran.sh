#!/bin/bash

module load gcc/13

cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/vecadd

make
./vecadd
make clean
