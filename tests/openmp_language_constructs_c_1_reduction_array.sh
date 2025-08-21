#!/bin/bash

module load rocm
module load amdclang
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C
cd 6_reduction_array/1_reduction_array
make
./reduction_array

make clean
