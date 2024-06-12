#!/bin/bash
export HSA_XNACK=1
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP-OpenMP/CXX/daxpy
make
./daxpy

make clean
