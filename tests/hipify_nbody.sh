#!/bin/bash

if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIPIFY/mini-nbody/cuda
hipify-perl -examine nbody-orig.cu

hipify-perl nbody-orig.cu > nbody-orig.cpp
hipcc -DSHMOO -I../ nbody-orig.cpp -o nbody-orig

./nbody-orig
