#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/jacobi

if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load openmpi

rm -rf build
mkdir build && cd build
cmake ..
make

#salloc -p LocalQ --gpus=2 -n 2 -t 00:10:00
mpirun -n 2 ./Jacobi_hip -g 2

cd ..
rm -rf build
