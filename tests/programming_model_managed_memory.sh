#!/bin/bash

module load amdclang

cd ${REPO_DIR}/ManagedMemory/Managed_Memory_Code
export HSA_XNACK=1
make gpu_code
./gpu_code

make clean
