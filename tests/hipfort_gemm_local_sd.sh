#!/bin/bash

module load amdclang
cd ${REPO_DIR}/HIPFort/hipgemm
make gemm_local_sd
./gemm_local_sd

make clean
