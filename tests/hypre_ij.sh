#!/bin/bash

# This test runs the ij test from HYPRE

# NOTE: this test assumes HYPRE has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/hypre_setup.sh

module purge

module load rocm openmpi hypre

HYPRE_VERSION=`cat $HYPRE_PATH/lib/cmake/HYPRE/HYPREConfigVersion.cmake | grep "set(PACKAGE_VERSION \"2"`
HYPRE_VERSION=`echo $HYPRE_VERSION | sed 's/set(PACKAGE_VERSION \"//g'`
HYPRE_VERSION=`echo $HYPRE_VERSION | sed 's/\")//g'`

git clone --branch v$HYPRE_VERSION https://github.com/hypre-space/hypre.git

pushd hypre/src/test

mpicc ij.c -o ij -I$HYPRE_PATH/include -L$HYPRE_PATH/lib -lHYPRE -lm


./ij -n 100 100 100 -pmis -keepT 1 -rlx 18 -exec_device -rap 1 -mod_rap2 1 -interptype 6 -solver 1 -agg_nl 1 -27pt -mxrs 0.9 -ns 2 -Pmx 8

popd

rm -rf hypre



