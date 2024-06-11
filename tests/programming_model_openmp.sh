#!/bin/bash
export HSA_XNACK=1

module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/ManagedMemory/OpenMP_Code
make openmp_code
./openmp_code

make clean
