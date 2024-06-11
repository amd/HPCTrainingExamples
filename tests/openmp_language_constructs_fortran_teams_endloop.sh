#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran
cd teams_endloop
make
./teams_endloop

make clean
