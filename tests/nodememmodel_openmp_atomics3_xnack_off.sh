#!/bin/bash

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/atomics_openmp

make arraysum3
export HSA_XNACK=0
./arraysum3

make clean

