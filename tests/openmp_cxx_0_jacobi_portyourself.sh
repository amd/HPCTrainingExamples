#!/bin/bash

module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/7_jacobi/0_jacobi_portyourself

make CC=$CXX
./Jacobi_omp
make clean
