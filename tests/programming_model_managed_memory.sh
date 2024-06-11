#!/bin/bash

module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/ManagedMemory/Managed_Memory_Code
export HSA_XNACK=1
make gpu_code
./gpu_code

make clean
