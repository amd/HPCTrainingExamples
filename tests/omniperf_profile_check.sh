#!/bin/bash

# This test checks that 
# omniperf profile runs

module purge

module load rocm
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `
pushd ${REPO_DIR}/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver1
mkdir build_for_test
cd build_for_test
cmake -DCMAKE_CXX_COMPILER=/opt/rocm-${ROCM_VERSION}/bin/amdclang++ -DCMAKE_C_COMPILER=/opt/rocm-${ROCM_VERSION}/bin/amdclang ..
make -j

result=`echo ${ROCM_VERSION} | awk '$1<=6.1.2'` && echo $result
module unload rocm

if [[ "${result}" ]]; then
   module load omniperf
   echo "loaded omniperf from AMD Research"
   echo " "
else
   module load rocm
   echo "loaded omniperf from ROCm"
   echo " "
fi   

export HSA_XNACK=1
omniperf profile -n v1 --no-roof -- ./GhostExchange -x 1 -y 1 -i 200 -j 200 -h 2 -t -c -I 100

cd ..
rm -rf build_for_test
popd
