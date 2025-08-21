#!/bin/bash

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/2_reduction/2_reduction_map

make
./reduction
make clean
