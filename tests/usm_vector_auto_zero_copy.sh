#!/bin/bash

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/USM/vector_add_auto_zero_copy

export HSA_XNACK=1
make
make run

make clean
