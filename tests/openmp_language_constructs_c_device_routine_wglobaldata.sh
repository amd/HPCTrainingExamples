#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/device_routine_wglobaldata
make
./device_routine

make clean
