#!/bin/bash

cd ${REPO_DIR}/HIP/jacobi

module load rocm
module load openmpi

mkdir build && cd build
cmake ..
make

#salloc -p LocalQ --gpus=2 -n 2 -t 00:10:00
mpirun -n 2 ./Jacobi_hip -g 2

cd ..
rm -rf build
