#!/bin/bash

module load aomp

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/C/1_saxpy/5_saxpy_map

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cp * ${BUILD_DIR}

cd ${BUILD_DIR}

make
./saxpy
