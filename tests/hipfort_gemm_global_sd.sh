#!/bin/bash

module load amdclang
cd ${REPO_DIR}/HIPFort/hipgemm
make gemm_global_sd
./gemm_global_sd

make clean
