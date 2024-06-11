#!/bin/bash

module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/atomics_openmp

make arraysum4
export HSA_XNACK=0
./arraysum4

make clean

