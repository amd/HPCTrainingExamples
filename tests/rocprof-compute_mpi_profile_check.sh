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

if command -v rocprof-compute &> /dev/null; then
    echo "rocprof-compute found at: $(which rocprof-compute)"
else
    echo "loading rocprofiler-compute module"
    module load rocprofiler-compute
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver1
rm -rf build_for_test
mkdir build_for_test
cd build_for_test
cmake -DCMAKE_CXX_COMPILER=`which amdclang++` -DCMAKE_C_COMPILER=`which amdclang` ..
make -j

export HSA_XNACK=1
rocprof-compute profile -n v1 --no-roof -- ./GhostExchange -x 1 -y 1 -i 200 -j 200 -h 2 -t -c -I 100

cd ..
rm -rf build_for_test
git checkout ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver1/GhostExchange.hip
popd

