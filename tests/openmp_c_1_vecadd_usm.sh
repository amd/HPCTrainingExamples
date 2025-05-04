#!/bin/bash

module load amdclang
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/3_vecadd/1_vecadd_usm

make
./vecadd
make clean
