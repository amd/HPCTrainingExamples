#!/bin/bash

module load amdclang
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/optimization/Allocations/2_opt_allocation

make
./opt_allocation
make clean
