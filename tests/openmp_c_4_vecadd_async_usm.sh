#!/bin/bash

module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/3_vecadd/4_vecadd_async_usm

export HSA_XNACK=1
make
./vecadd
make clean
