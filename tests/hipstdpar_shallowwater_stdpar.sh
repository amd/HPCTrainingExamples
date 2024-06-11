#!/bin/bash

export HSA_XNACK=1
module load amdclang

cd ${REPO_DIR}/HIPStdPar/CXX/ShallowWater_StdPar

make
#export AMD_LOG_LEVEL=3
./ShallowWater

make clean
