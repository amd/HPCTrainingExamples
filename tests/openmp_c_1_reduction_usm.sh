#!/bin/bash

module load rocm
module load amdclang
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/3_reduction/1_reduction_solution_usm

make
./reduction
make clean
