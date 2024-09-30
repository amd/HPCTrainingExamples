#!/bin/bash

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP-OpenMP/CXX/saxpy_openmp_offload
make

make clean
