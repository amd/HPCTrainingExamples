#!/bin/bash

module load amdflang-new
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/5_device_routines/device_routine_with_interface/5_device_routine_enter_data
make
./device_routine

make clean
