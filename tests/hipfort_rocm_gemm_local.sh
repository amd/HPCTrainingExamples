#!/bin/bash

module load amdclang
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPFort/hipgemm
make gemm_local
./gemm_local

make clean
