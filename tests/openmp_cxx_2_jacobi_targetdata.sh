#!/bin/bash

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/8_jacobi/2_jacobi_targetdata

make CC=$CXX
./Jacobi_omp
make clean
