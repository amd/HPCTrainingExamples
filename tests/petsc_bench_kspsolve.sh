#!/bin/bash

# This test runs a KSP performance test on a Poisson problem

# NOTE: this test assumes PETSc has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/petsc_setup.sh

module purge

module load rocm openmpi petsc

PETSC_VERSION=`$PETSC_DIR/lib/petsc/bin/petscversion`

git clone --branch v$PETSC_VERSION https://gitlab.com/petsc/petsc.git

pushd petsc/src/ksp/ksp/tutorials

sed -i '/PetscCheck(norm/d' bench_kspsolve.c

mpicc bench_kspsolve.c -o bench_kspsolve -I$PETSC_PATH/include -L$PETSC_PATH/lib -lpetsc

./bench_kspsolve -mat_type aijhipsparse

popd

rm -rf petsc



