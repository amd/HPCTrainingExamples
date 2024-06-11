#!/bin/bash

module load amdclang
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/device_routine_wdynglobaldata
make
./device_routine

make clean
