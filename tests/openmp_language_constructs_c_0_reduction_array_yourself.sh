#!/bin/bash

module load amdclang
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C
cd 6_reduction_array/0_reduction_array_yourself
make
./reduction_array

make clean
