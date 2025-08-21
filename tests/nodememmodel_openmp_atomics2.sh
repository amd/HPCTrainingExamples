#!/bin/sh

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/atomics_openmp

make arraysum2
HSA_XNACK=1
./arraysum2

make clean

