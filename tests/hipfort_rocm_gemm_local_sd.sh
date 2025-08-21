#!/bin/bash

module load rocm
module load amdclang
module load hipfort
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPFort/hipgemm
make clean
make gemm_local_sd
./gemm_local_sd
make clean
