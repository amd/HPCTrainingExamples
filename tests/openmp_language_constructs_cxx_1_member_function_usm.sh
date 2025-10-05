#!/bin/bash

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/6_device_routines/1_member_function/1_member_function
export HSA_XNACK=1
make
./bigscience
make clean
