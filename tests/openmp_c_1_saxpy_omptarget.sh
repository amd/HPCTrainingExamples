#!/bin/bash

module load rocm
module load amdclang
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/1_saxpy/1_saxpy_omptarget

make
./saxpy
make clean
