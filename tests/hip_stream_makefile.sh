#!/bin/bash

module load rocm

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/hip-stream

make stream
./stream
make clean
