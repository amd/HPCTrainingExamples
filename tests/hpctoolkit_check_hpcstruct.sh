#!/bin/bash

# This test checks that hpcstruct
# runs without errors

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

module load hpctoolkit
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/HIP/Stream_Overlap/0-Orig/
rm -rf build_for_test
mkdir build_for_test; cd build_for_test
cmake ../
make -j

hpcrun -e CPUTIME -e gpu=amd -t ./compute_comm_overlap 2
hpcstruct hpctoolkit-compute_comm_overlap-measurements*
ls hpctoolkit-compute_comm_overlap-measurements*

cd ..
rm -rf build_for_test

popd

