#!/bin/bash

module load gcc/13

cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/freduce

make
./freduce
make clean
