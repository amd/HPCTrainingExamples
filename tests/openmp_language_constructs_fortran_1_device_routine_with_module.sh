#!/bin/bash

module load rocm
module load amdflang-new
export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/6_device_routines/device_routine_with_module/1_device_routine_with_module
make
./device_routine

make clean
