#!/bin/bash

module load amdclang
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/5_device_routines/2_member_function_external/2_member_function_external_data
make
./bigscience

make clean
