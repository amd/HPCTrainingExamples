#!/bin/sh

module load rocm
module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/atomics_openmp

make arraysum3
./arraysum3

make clean

