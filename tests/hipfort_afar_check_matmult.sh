#!/bin/bash

module load rocm
module load amdflang-new
export HIPFORT_PATH=$AFAR_PATH 
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPFort/matmult
make clean
make
./matmult_hipfort
make clean
