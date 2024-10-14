#!/bin/bash

module load amdclang
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/0_Intro
make target_data_update
./target_data_update

make clean
