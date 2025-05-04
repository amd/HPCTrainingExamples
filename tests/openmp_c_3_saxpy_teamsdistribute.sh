#!/bin/bash

module load amdclang
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/1_saxpy/3_saxpy_teamsdistribute

make
./saxpy
make clean
