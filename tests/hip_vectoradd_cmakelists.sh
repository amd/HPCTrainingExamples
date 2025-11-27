#!/bin/bash

if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/vectorAdd

rm -rf build
mkdir build && cd build
cmake ..
make
./vectoradd
cd ..
rm -rf build
