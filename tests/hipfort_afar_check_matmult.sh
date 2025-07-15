#!/bin/bash

module load amdflang-new
export HIPFORT_PATH=$AFAR_PATH 
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPFort/matmult
make
./matmult_hipfort

make clean
