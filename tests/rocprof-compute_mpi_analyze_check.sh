#!/bin/bash

# This test checks that
# rocprofiler-compute (formerly omniperf) analyze with mpi runs

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load openmpi

module load rocprofiler-compute &> /dev/null

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver1
rm -rf build_for_test
mkdir build_for_test
cd build_for_test
cmake -DCMAKE_CXX_COMPILER=`which amdclang++` -DCMAKE_C_COMPILER=`which amdclang` ..
make -j

export HSA_XNACK=1
rocprof-compute profile -n v1 --no-roof -- ./GhostExchange -x 1 -y 1 -i 200 -j 200 -h 2 -t -c -I 100
sed -i "/dim3 grid((isize+63)\/64, (jsize+3)\/4, 1);/c\   dim3 grid((isize+255)\/256, (jsize+3)\/4, 1);" ../GhostExchange.hip
sed -i "/dim3 block(64, 4, 1);/c\   dim3 block(256, 4, 1);" ../GhostExchange.hip
rocprof-compute profile -n v2 --no-roof -- ./GhostExchange -x 1 -y 1 -i 200 -j 200 -h 2 -t -c -I 100
rocprof-compute analyze -p workloads/v1/* -p workloads/v2/* --block 7.1.0 7.1.1 7.1.2 7.1.0: Grid size 7.1.1: Workgroup size 7.1.2: Total Wavefronts

cd ..
rm -rf build_for_test
git checkout ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver1/GhostExchange.hip
popd

