#!/bin/bash

module load rocm
module load amdclang 
module load hipfort
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPFort/matmult
make clean
make
./matmult_hipfort
make clean
