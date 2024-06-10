#!/bin/bash

module load amdclang

cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/freduce

make
./freduce
make clean
