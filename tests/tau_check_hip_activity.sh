#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/jacobi

module purge
module load tau

make

mpirun -n 2 tau_exec -T rocm -ebs   ./Jacobi_hip -g 2 1
pprof


