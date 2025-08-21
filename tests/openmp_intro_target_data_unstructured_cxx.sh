#!/bin/bash

module load rocm
module load amdclang
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Intro
make target_data_unstructured
./target_data_unstructured

make clean
