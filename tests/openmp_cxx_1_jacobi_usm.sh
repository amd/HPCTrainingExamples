#!/bin/bash

module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/7_jacobi/1_jacobi_usm

export HSA_XNACK=1
make
./jacobi
make clean
