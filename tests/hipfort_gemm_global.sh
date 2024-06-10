#!/bin/bash

module load amdclang
cd ${REPO_DIR}/HIPFort/hipgemm
make gemm_global
./gemm_global

make clean
