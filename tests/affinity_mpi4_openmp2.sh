#!/bin/bash

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load gcc openmpi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRC_DIR=${REPO_DIR}/tests/hello_mpi_omp
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cp -r ${SRC_DIR}/* ${BUILD_DIR}/
cd ${BUILD_DIR}
make

OMP_NUM_THREADS=2 OMP_PROC_BIND=close mpirun -np 4 -mca btl ^openib --map-by L3cache ./hello_mpi_omp
