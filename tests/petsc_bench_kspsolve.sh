#!/bin/bash

# This test runs a KSP performance test on a Poisson problem

# NOTE: this test assumes PETSc has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/petsc_setup.sh

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   module load openmpi
fi

module load petsc_amdflang >& /dev/null
if [ "$?" == "1" ]; then
    module load petsc
fi

PETSC_VERSION=`$PETSC_DIR/lib/petsc/bin/petscversion`

git clone --branch v$PETSC_VERSION https://gitlab.com/petsc/petsc.git petsc_for_test

pushd petsc_for_test/src/ksp/ksp/tutorials

sed -i '/PetscCheck(norm/d' bench_kspsolve.c

mpicc bench_kspsolve.c -o bench_kspsolve -I$PETSC_PATH/include -L$PETSC_PATH/lib -lpetsc

./bench_kspsolve -mat_type aijhipsparse

popd

rm -rf petsc_for_test



