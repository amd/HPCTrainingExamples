#!/bin/bash

# This test checks that we can run a PETSc Fortran example correctly

# NOTE: this test assumes PETSc has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/petsc_setup.sh


if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load openmpi petsc_amdflang

PETSC_VERSION=`$PETSC_DIR/lib/petsc/bin/petscversion`

git clone --branch v$PETSC_VERSION https://gitlab.com/petsc/petsc.git petsc_for_test

pushd petsc_for_test/src/snes/tests

mpifort ex21f.F90 -o ex21f -I$PETSC_PATH/include -L$PETSC_PATH/lib -lpetsc

./ex21f ksp_gmres_cgs_refinement_type refine_always   -snes_monitor -snes_converged_reason -snes_view -pc_type lu

popd

rm -rf petsc_for_test



