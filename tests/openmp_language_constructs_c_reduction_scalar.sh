#!/bin/bash

module load rocm
module load amdclang
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C
cd 4_reduction_scalars/1_reduction_scalars
make
./reduction_scalar

make clean
