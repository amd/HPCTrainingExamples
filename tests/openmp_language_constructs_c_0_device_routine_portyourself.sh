#!/bin/bash

module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/5_device_routines/1_device_routine/0_device_routine_portyourself

make
./device_routine
make clean
