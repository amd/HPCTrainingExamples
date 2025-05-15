#!/bin/bash

module load amdclang
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/5_device_routines/device_routine_with_interface/0_device_routine_portyourself
make
./device_routine

make clean
