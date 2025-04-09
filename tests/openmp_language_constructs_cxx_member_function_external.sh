#!/bin/bash

module load amdclang
external HSA_XNACK=1
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/5_device_routines/2_member_function_external/1_member_function_external
make
./bigscience

make clean
