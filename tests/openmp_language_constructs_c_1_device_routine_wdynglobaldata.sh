#!/bin/bash

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/6_device_routines/3_device_routine_wdynglobaldata/1_device_routine_wdynglobaldata

make
./device_routine
make clean
