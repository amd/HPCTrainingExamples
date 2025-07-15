#!/bin/bash

module load hipfort
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPFort/hipgemm
make gemm_global_sd
./gemm_global_sd

make clean
