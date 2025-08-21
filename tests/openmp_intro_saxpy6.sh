#!/bin/bash

module load rocm
module load amdclang
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Intro
make saxpy6
./saxpy6

make clean
