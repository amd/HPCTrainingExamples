#!/bin/bash

export HSA_XNACK=1
module load amdclang

cd ${REPO_DIR}/HIPStdPar/CXX/saxpy_transform

make
export AMD_LOG_LEVEL=3
./saxpy

make clean
