#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/jacobi

module purge
module load rocm
module load tau

make

tau_exec -T rocm -ebs   ./Jacobi_hip -g 1 1
pprof


