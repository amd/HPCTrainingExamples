#!/bin/bash

# This test runs a KSP performance test on a Poisson problem

# NOTE: this test assumes PETSc has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/petsc_setup.sh

PETSC_MODULE="petsc"

usage()
{
    echo ""
    echo "--help : prints this message"
    echo "--petsc-module : specifies which petsc module to load"
    echo ""
    exit
}

send-error()
{
    usage
    echo -e "\nError: ${@}"
    exit 1
}

reset-last()
{
   last() { send-error "Unsupported argument :: ${1}"; }
}

n=0
while [[ $# -gt 0 ]]
do
   case "${1}" in
      "--petsc-module")
          shift
          PETSC_MODULE=${1}
          reset-last
          ;;
     "--help")
          usage
          ;;
      "--*")
          send-error "Unsupported argument at position $((${n} + 1)) :: ${1}"
          ;;
      *)
         last ${1}
         ;;
   esac
   n=$((${n} + 1))
   shift
done


module purge

module load rocm openmpi $PETSC_MODULE

PETSC_VERSION=`$PETSC_DIR/lib/petsc/bin/petscversion`

git clone --branch v$PETSC_VERSION https://gitlab.com/petsc/petsc.git petsc_for_test

pushd petsc_for_test/src/ksp/ksp/tutorials

sed -i '/PetscCheck(norm/d' bench_kspsolve.c

mpicc bench_kspsolve.c -o bench_kspsolve -I$PETSC_PATH/include -L$PETSC_PATH/lib -lpetsc

./bench_kspsolve -mat_type aijhipsparse

popd

rm -rf petsc_for_test



