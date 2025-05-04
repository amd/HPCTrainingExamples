#!/bin/bash

module load amdclang

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/3_vecadd/0_vecadd_portyourself

make
./vecadd
make clean
