#!/bin/bash

# This test runs the ij test from HYPRE

# NOTE: this test assumes HYPRE has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/hypre_setup.sh


module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load openmpi hypre

HYPRE_VERSION=`cat $HYPRE_PATH/lib/cmake/HYPRE/HYPREConfigVersion.cmake | grep "set(PACKAGE_VERSION \"3"`
if [[ "$HYPRE_VERSION" == "" ]]; then
   HYPRE_VERSION=`cat $HYPRE_PATH/lib/cmake/HYPRE/HYPREConfigVersion.cmake | grep "set(PACKAGE_VERSION \"2"`
fi
HYPRE_VERSION=`echo $HYPRE_VERSION | sed 's/set(PACKAGE_VERSION \"//g'`
HYPRE_VERSION=`echo $HYPRE_VERSION | sed 's/\")//g'`

git clone --branch v$HYPRE_VERSION https://github.com/hypre-space/hypre.git

pushd hypre/src/test

mpicc ij.c -o ij -I$HYPRE_PATH/include -L$HYPRE_PATH/lib -lHYPRE -lm


./ij -n 25 25 25 -pmis -keepT 1 -rlx 18 -exec_device -rap 1 -mod_rap2 1 -interptype 6 -solver 1 -agg_nl 1 -27pt -mxrs 0.9 -ns 2 -Pmx 8

popd

rm -rf hypre



