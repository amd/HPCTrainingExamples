#!/bin/bash

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/6_device_routines/2_device_routine_wglobaldata/0_device_routine_wglobaldata_portyourself

make
./device_routine
make clean
