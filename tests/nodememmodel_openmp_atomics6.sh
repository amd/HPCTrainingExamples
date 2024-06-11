#!/bin/bash

module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/atomics_openmp

make arraysum6
export HSA_XNACK=1
./arraysum6

make clean

