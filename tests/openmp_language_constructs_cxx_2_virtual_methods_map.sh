#!/bin/bash

module load amdclang
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/CXX/5_device_routines/3_virtual_methods/2_virtual_methods_map
make
./bigscience

make clean
