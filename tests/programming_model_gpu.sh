#!/bin/bash

module load amdclang

cd ${REPO_DIR}/ManagedMemory/GPU_Code
make gpu_code
./gpu_code

make clean
