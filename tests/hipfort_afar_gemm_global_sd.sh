#!/bin/bash

module load amdflang-new
export HIPFORT_PATH=$AFAR_PATH
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPFort/hipgemm
make gemm_global_sd
./gemm_global_sd

make clean
