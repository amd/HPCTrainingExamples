#!/bin/bash

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/4_reduction_scalars/0_reduction_scalars_portyourself

make
./reduction_scalar
make clean
