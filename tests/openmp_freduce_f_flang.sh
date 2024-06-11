#!/bin/bash

module load clang/15

cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/freduce

make
./freduce
make clean
