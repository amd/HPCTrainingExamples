#!/bin/bash

module load rocm
module load amdclang
export HSA_XNACK=1
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/5_device_routines/3_virtual_methods/1_virtual_methods
make
./bigscience

make clean
