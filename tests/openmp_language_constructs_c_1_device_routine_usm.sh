#!/bin/bash

module load amdclang
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/5_device_routines/1_device_routine/1_device_routine_usm

make
./device_routine
make clean
