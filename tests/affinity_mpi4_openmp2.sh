#!/bin/bash

module load gcc rocm openmpi

git clone https://code.ornl.gov/olcf/hello_mpi_omp.git
cd hello_mpi_omp
sed -i -e '/COMP/s/cc/mpicc/' Makefile
make

OMP_NUM_THREADS=2 OMP_PROC_BIND=close mpirun -np 4 -mca btl ^openib --map-by L3cache ./hello_mpi_omp

cd ..
rm -rf hello_mpi_omp
