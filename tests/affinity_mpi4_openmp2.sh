#!/bin/bash

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load gcc openmpi

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd ${BUILD_DIR}

git clone https://code.ornl.gov/olcf/hello_mpi_omp.git
cd hello_mpi_omp
sed -i -e '/COMP/s/cc/mpicc/' Makefile
make

OMP_NUM_THREADS=2 OMP_PROC_BIND=close mpirun -np 4 -mca btl ^openib --map-by L3cache ./hello_mpi_omp
