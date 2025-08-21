#!/bin/bash

module load rocm
module load hipfort_from_source
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPFort/hipgemm
make clean
make gemm_global
./gemm_global
make clean
