#!/bin/bash

# This test checks that hpcprof
# runs without errors



module load hpctoolkit
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/HIP/Stream_Overlap/0-Orig/
rm -rf build_for_test
mkdir build_for_test; cd build_for_test
cmake ../
make -j

hpcrun -e CPUTIME -e gpu=amd -t ./compute_comm_overlap 2
hpcstruct hpctoolkit-compute_comm_overlap-measurements/
hpcprof hpctoolkit-compute_comm_overlap-measurements/
ls hpctoolkit-compute_comm_overlap-database

cd ..
rm -rf build_for_test

popd

