#!/bin/bash

module load amdclang
cd ${REPO_DIR}/HIPFort/hipgemm
make gemm_local
./gemm_local

make clean
