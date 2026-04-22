#!/bin/bash

module load amd-gcc
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran
cd complex_saxpy

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cp * ${BUILD_DIR}

cd ${BUILD_DIR}

make
./complex_saxpy
