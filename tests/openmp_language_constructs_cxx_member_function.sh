#!/bin/bash

module load amdclang
export HSA_XNACK=1
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/5_device_routines/1_member_function/1_member_function
make
./bigscience

make clean
