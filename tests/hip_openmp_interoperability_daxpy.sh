#!/bin/bash
export HSA_XNACK=1
module load amdclang

cd ${REPO_DIR}/HIP-OpenMP/CXX/daxpy
make
./daxpy

make clean
